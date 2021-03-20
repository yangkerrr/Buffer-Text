import copy
import cv2
import os
import torch.utils.data as data
import scipy.io as io
import numpy as np
import math
import torch
from PIL import Image
from util.config import config as cfg
from skimage.draw import polygon as drawpoly
from util.misc import find_bottom, find_long_edges, split_edge_seqence, \
    norm2, vector_cos, vector_sin,shrink
Image.MAX_IMAGE_PIXELS = None


def pil_load_img(path):
    image = Image.open(path)
    image = np.array(image)
    return image


class TextInstance(object):

    def __init__(self, points, orient, text):
        self.orient = orient
        self.text = text

        remove_points = []

        if len(points) > 4:
            # remove point if area is almost unchanged after removing it
            ori_area = cv2.contourArea(points)
            for p in range(len(points)):
                # attempt to remove p
                index = list(range(len(points)))
                index.remove(p)
                area = cv2.contourArea(points[index])
                if np.abs(ori_area - area) / ori_area < 0.017 and len(points) - len(remove_points) > 4:
                    remove_points.append(p)
            self.points = np.array([point for i, point in enumerate(points) if i not in remove_points])
        else:
            self.points = np.array(points)


    def __repr__(self):
        return str(self.__dict__)

    def __getitem__(self, item):
        return getattr(self, item)


class TextDataset(data.Dataset):

    def __init__(self, transform):
        super().__init__()
        self.device = torch.device("cuda")
        self.transform = transform



    def make_text_region(self, image, polygons):

        tr_mask = np.zeros(image.shape[:2], np.uint8)
        train_mask = np.ones(image.shape[:2], np.uint8)

        for polygon in polygons:
            cv2.fillPoly(tr_mask, [polygon.points.astype(np.int32)], color=(1,))
            if polygon.text == '#':
                cv2.fillPoly(train_mask, [polygon.points.astype(np.int32)], color=(0,))
        return tr_mask, train_mask
    
    def find_min_radii(self,pts):
        
        bottom = find_bottom(pts)
        e1, e2 = find_long_edges(pts, bottom)  # find two long edge sequence
        inner_points1 = split_edge_seqence(pts, e1, 16)
        inner_points2 = split_edge_seqence(pts, e2, 16)
        inner_points2 = inner_points2[::-1]
        center_points = (inner_points1 + inner_points2) / 2  # disk center
        radii = norm2(inner_points1 - center_points, axis=1)
        return inner_points1,inner_points2,center_points
    
    def SandE(self,pts,distance):
        
        pco = pyclipper.PyclipperOffset()
        pco.AddPath(pts, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        poly = np.array(pco.Execute(-distance))
        return poly
    
    def fill_polygon(self,mask, polygon, value):
        """
        fill polygon in the mask with value
        :param mask: input mask
        :param polygon: polygon to draw
        :param value: fill value
        """
        rr, cc = drawpoly(polygon[:, 1], polygon[:, 0],shape=(512, 512))
        mask[rr, cc] = value
    
    def expand(self,points):
        points_11 = points[0] + 0.5*(points[0]-points[1])
        points_22 = points[-1] + 0.5*(points[-1] - points[-2])
        points = np.insert(points, 0, points_11,axis=0)
        points = np.append(points,[points_22],axis=0)
        return points

    def make_shrink(self,image,m):
        shrink_mask = np.zeros(image.shape[:2],dtype=np.uint8)
        shrink_train_mask = np.ones(image.shape[:2],dtype=np.uint8)
        for polygon in m:
            pts = polygon.points.astype(np.int32)
            inner_points1,inner_points2,center_points = self.find_min_radii(pts)
            for i in range(4, 11):
                c1 = center_points[i]
                c2 = center_points[i + 1]
                top1 = inner_points1[i]
                top2 = inner_points1[i + 1]
                bottom1 = inner_points2[i]
                bottom2 = inner_points2[i + 1]
                p1 = c1 + (top1 - c1)/3
                p2 = c1 + (bottom1 - c1)/3
                p3 = c2 + (bottom2 - c2)/3
                p4 = c2 + (top2 - c2)/3
                polygon1 = np.stack([p1, p2, p3, p4])
                self.fill_polygon(shrink_mask, polygon1, 1)
                if polygon.text == '#':
                    self.fill_polygon(shrink_train_mask, polygon1, 0)
        return shrink_mask,shrink_train_mask
    
    def get_one_hot(self,label, N):
        size = list(label.size())
        label = label.view(-1)
        ones = torch.sparse.torch.eye(N)
        ones = ones.index_select(0, label)
        size.append(N)
        return ones.view(*size)
    
    def fill_polygon(self,mask, polygon, value):
        h,w = mask.shape
        rr, cc = drawpoly(polygon[:, 1], polygon[:, 0])
        rr[rr >= h] = h-1
        cc[cc>=w] = w-1
        mask[rr, cc] = value
    
    #shrink
    def shrinkpoly(self,mask,inner_points1, inner_points2,center_points,rate):
        h,w = mask.shape
        mm = []
        nn = []
        for i in range(4, 12):
            c1 = center_points[i]
            c2 = center_points[i + 1]
            top1 = inner_points1[i]
            top2 = inner_points1[i + 1]
            bottom1 = inner_points2[i]
            bottom2 = inner_points2[i + 1]
            p1 = c1 + (top1 - c1)*rate/2
            p2 = c1 + (bottom1 - c1)*rate/2
            p3 = c2 + (bottom2 - c2)*rate/2
            p4 = c2 + (top2 - c2)*rate/2
            polygon1 = np.stack([p1, p2, p3, p4])
            if (i == 4):
                mm.append(p1)
                mm.append(p2)
            elif(i == 11):
                mm.append(p3)
                nn.append(p4)
            else:
                mm.append(p2)
                nn.append(p1)
        nn=nn[: :-1]
        for i in nn:
            mm.append(i)
        xs = [i[0] for i in mm]
        ys = [i[1] for i in mm]
        newp = np.stack([xs,ys]).T.astype(np.int32)
        cv2.fillPoly(mask, [newp], (255,255,255))
    
    #Polygon expansion 
    def expandpoly(self,mask,inner_points1, inner_points2,center_points):
        h,w = mask.shape
        inner_points1 = self.expand(inner_points1)
        inner_points2 = self.expand(inner_points2)
        center_points = self.expand(center_points)
        mm = []
        nn = []
        for i in range(0, 18):
            c1 = center_points[i]
            c2 = center_points[i + 1]
            top1 = inner_points1[i]
            top2 = inner_points1[i + 1]
            bottom1 = inner_points2[i]
            bottom2 = inner_points2[i + 1]
            p1 = c1 + (top1 - c1)*1.5
            p2 = c1 + (bottom1 - c1)*1.5
            p3 = c2 + (bottom2 - c2)*1.5
            p4 = c2 + (top2 - c2)*1.5
            polygon1 = np.stack([p1, p2, p3, p4])
            if (i == 0):
                mm.append(p1)
                mm.append(p2)
            elif(i == 17):
                mm.append(p3)
                nn.append(p4)
            else:
                mm.append(p2)
                nn.append(p1)
        nn=nn[: :-1]
        for i in nn:
            mm.append(i)
        xs = [i[0] for i in mm]
        ys = [i[1] for i in mm]
        newp = np.stack([xs,ys]).T.astype(np.int32)
        cv2.fillPoly(mask, [newp], (255,255,255))
    
    #Three-category label with buffer region
    def make_label(self,image,polygons):
        heatmap = np.zeros(image.shape[:2], np.float32)
        distance = np.zeros((int(image.shape[0]),int(image.shape[1])), np.float32)
        mm =np.zeros(image.shape[:2], np.uint8) 
        for polygon in polygons:
            try:
                pts = polygon.points.astype(np.int32)
                heatmap1=heatmap.copy()
                inner_points1, inner_points2,center_points = self.find_min_radii(pts)
                mask = np.zeros(image.shape[:2], np.uint8)
                mask1 = np.zeros(image.shape[:2], np.uint8)
                self.expandpoly(mask1,inner_points1, inner_points2,center_points)
                cv2.fillPoly(mask,[pts],255)
                cv2.fillPoly(mm,[pts],255)
                heatmap[mask == 255] = heatmap[mask == 255] +1
                self.expandpoly(mask,inner_points1, inner_points2,center_points)
                heatmap[mask == 255] = heatmap[mask == 255] +1
                heatmap[mask1*heatmap1!=0] = 1
                heatmap[heatmap>2] = 2
            except:
                #print("error")
                continue
        distance = cv2.distanceTransform(mm, cv2.DIST_L2, 3)
        heatmap = heatmap.astype(np.uint8)
        return heatmap,distance



    def get_training_data(self, image, polygons, image_id, image_path):

        H, W, _ = image.shape
        
        if self.transform:
            image, polygons = self.transform(image, copy.copy(polygons))
        
        tr_mask, train_mask = self.make_text_region(image, polygons)

        label,distance = self.make_label(image,polygons)
        label = torch.LongTensor(label)
        label = label.to(self.device)
        label = self.get_one_hot(label,3)

        image = image.transpose(2, 0, 1)
        points = np.zeros((cfg.max_annotation, cfg.max_points, 2))
        length = np.zeros(cfg.max_annotation, dtype=int)

        meta = {
            'image_id': image_id,
            'image_path': image_path,
            'annotation': points,
            'n_annotation': length,
            'Height': H,
            'Width': W
        }


        
        
        return image,label,tr_mask,train_mask,distance,meta


    def get_training_data1(self, image,image_id):

        H, W, _ = image.shape
        
        #transform the image
        if self.transform:
            image,_= self.transform(image)
        
        
        #获得
        
        image = image.transpose(2, 0, 1)

        meta = {
        'image_id': image_id,
        'Height': H,
        'Width': W
        }
        
        return image,meta