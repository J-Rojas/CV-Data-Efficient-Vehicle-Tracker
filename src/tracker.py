from src.detector import SegmentationLightning
import numpy as np
import cv2
import torch
import os.path    
import glob
import pandas as pd
import argparse    
from numpy.lib.stride_tricks import sliding_window_view
from .detector import SegmentationLightning
from .tools import slice_from_bbox, patch_matching_cross_correlation, find_best_correspondence, iou, rescale_bboxes, torch_to_cv2_image, overlay_mask_on_image, size_from_bbox
from skimage import measure
from bisect import bisect_left
from typing import Tuple
from .loader import VehicleSegmentationAugmentedEvaluationDataset, DataLoader, validation_images_dir, labels_dir, feature_extractor

OUTPUT_SIZE = [272, 640]
OUTPUT_SIZE_CV = [640, 272]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def convert_image(normalized_img):
    
    # Example mean and std values, adjust these according to your feature extractor's config.
    mean = np.array([0.485, 0.456, 0.406])  # e.g., ImageNet mean
    std = np.array([0.229, 0.224, 0.225])   # e.g., ImageNet std

    # If your normalized image is in (C, H, W), first convert to (H, W, C) for easier processing.
    if normalized_img.shape[0] == 3:
        normalized_img = np.transpose(normalized_img, (1, 2, 0))

    # Undo the normalization:
    # Make sure the image is float32
    recovered = (normalized_img * std) + mean

    # Optionally, if you want 8-bit values:
    recovered_uint8 = np.clip(recovered * 255, 0, 255).astype(np.uint8)

    return recovered_uint8

class TrackingObject():
    
    def __init__(self, id) -> None:
        self.num_id = id
        self.frame_pixels = {}
        self.regions = {}
        self.pred_mask = {}

    def to_json(self):
        return {
            "num_id": self.num_id,
            "frame_pixels": self.frame_pixels,
            "regions": self.regions,
            "pred_mask": self.pred_mask
        }

    def from_json(self, data):
        self.num_id = data["num_id"]
        self.frame_pixels = data["frame_pixels"]
        self.regions = data["regions"]
        self.pred_mask = data["pred_mask"]

    def best_scores(self, scores):
        return scores[1:]

    def compare(self, idx, pixels, use_containing_region=False):        
        pixel_regions = [self.get_containing_pixels(idx)] if use_containing_region else self.frame_pixels.get(idx) 
        if pixel_regions:
            """use_convolve=False, pad=True, unnormalized=False"""
            scores = [self.best_scores(patch_matching_cross_correlation(pixel_region, pixels)) for pixel_region in pixel_regions] 
            # print(scores)
            high_scores = list(map(lambda x: x[0], scores))
            idx = np.argmax(high_scores)
            pos = scores[idx][1]
            max_score = scores[idx][0]            
            return max_score, pos if use_containing_region else np.concatenate([idx, pos]) # here, the pposition can be 3d dimensions, (region H W)
        return np.array([0, 0]), np.array([0, 0])

    def get_template(self, idxes):
        """construct a templated view of the object within a window"""        
        frame_pixels = [self.frame_pixels[i] for i in idxes]
        sizes, min_idx, max_idx = self.get_sizes_as_array(idxes)
        max_size = np.max(sizes, axis=0)
        template = np.zeros(max_size)
        
        # construct a rough representation of the object across frames
        for im in frame_pixels:
            start_y, start_x = (template.shape[0] - im.shape[0]) // 2, (template.shape[1] - im.shape[1]) // 2  
            template[start_y:start_y+im.shape[0], start_x:start_x+im.shape[1]] += im

        template = template.mean()

        return template

    def overlaps(self, idx, bbox, history=1):
        """ return true of the idx frame region overlaps the bbox """
        boxes = []
        for i in range(history + 1):
            regions = self.regions.get(idx - i, None)
            if regions:
                boxes.extend(regions)
        return np.mean([iou(bbox, box) for box in boxes]) if boxes else 0.


    def update(self, idx, pixels, region, pred_mask=None):
        fp = self.frame_pixels[idx] = self.frame_pixels.get(idx, [])
        re = self.regions[idx] = self.regions.get(idx, [])
        pd = self.pred_mask[idx] = self.pred_mask.get(idx, [])        
        fp.append(pixels)
        re.append(region)
        if pred_mask is not None:
            assert(pred_mask.shape == size_from_bbox(region))
            pd.append(pred_mask)

    def clear(self, idx):
        self.frame_pixels[idx] = []
        self.regions[idx] = []
        self.pred_mask[idx] = []

    def get_pos_as_array(self):
        min_idx = min(self.regions.keys())
        max_idx = max(self.regions.keys())
        arr = np.zeros((max_idx - min_idx, 2))
        for i, r in enumerate(self.regions.values()):
            arr[i,:] = [(r[0] + r[2]) / 2, (r[1] - r[3]) / 2]
        return arr, min_idx, max_idx

    def get_containing_region(self, idx):
        region_list = self.regions.get(idx)
        if region_list is None:
            return None
        region_arr = np.array(region_list)
        min_region = region_arr.min(axis=0)
        max_region = region_arr.max(axis=0)
        return [min_region[0], min_region[1], max_region[2],  max_region[3]]

    def _merge_pixel_regions(self, idx, pixels_list):        
        if pixels_list is None:
            return None
        # if there is only one region, then return 
        if len(pixels_list) == 1:
            return pixels_list[0]
        # otherwise merge the pixels together into one region
        region = self.get_containing_region(idx)
        region_list = self.regions.get(idx)
        pixels = None
        if region and region_list:
            pixels = np.zeros((region[2] - region[0] + 1, region[3] - region[1] + 1))
            # merge pixel regions
            for r, p in zip(region_list, pixels_list):
                pixels[r[0] - region[0]:r[2] - region[0] + 1,r[1] - region[1]:r[3] - region[1] + 1] = p

        assert(pixels.shape == size_from_bbox(region))
        return pixels

    def get_containing_pixels(self, idx):
        return self._merge_pixel_regions(idx, self.frame_pixels.get(idx))

    def get_containing_mask(self, idx):
        return self._merge_pixel_regions(idx, self.pred_mask.get(idx))
        
    def get_sizes_as_array(self, idxes=None):        
        arr = np.zeros((len(self.regions), 2))
        for i, region_list in enumerate(self.regions.values()):
            region_arr = np.array(region_list)
            min_region = region_arr.min(axis=0)
            max_region = region_arr.max(axis=0)
            arr[i,:] = [abs(min_region[0] - max_region[2]), abs(min_region[1] - max_region[3])]
        
        return arr

    def region_offset(self, reference_frame, target_frame):
        pixels = self.get_containing_pixels(target_frame)        
        return self.compare(reference_frame, pixels, use_containing_region=True)[1] if pixels is not None else (0, 0)

    def find_best_fit(self, reference_frame, target_frame):
        template = self.get_containing_pixels(target_frame)
        image = self.get_containing_pixels(reference_frame)

        # use padding only if the sizes are very different for performance
        diff = np.abs(np.array(image.shape) - np.array(template.shape))
        
        use_pad = np.sum(np.max(diff, axis=0) / np.max(image.shape, axis=0) > 0.25) > 0

        results = find_best_correspondence(image, template, use_pad=use_pad)

        high_scores = results[0]
        idx = np.argmax(high_scores)
            
        return results[1][idx]



class Tracker():

    SIMILARITY_THRESHOLD = 0.2
    SLIDING_WINDOW_SIZE = 3
    FILTER_SLIDING_WINDOW_SIZE = 5
    REGION_FILTER_WINDOW_SIZE = 5
    REGION_FILTER_THRESHOLD = 0.8
    REGION_MIN_DIMENSION = 15
    IOU_THRESHOLD = 0.01

    def __init__(self, detector: SegmentationLightning) -> None:
        self.detector = detector
        self.objects = []

    def to_json(self):
        return {
            "objects": [obj.to_json() for obj in self.objects]            
        }

    def from_json(self, data):
        self.objects = []
        for item in data["objects"]:
            obj= TrackingObject(0)
            obj.from_json(item)
            self.objects.append(obj)

    def get_detection_masks(self, im):
        im, preds, logits, loss = self.detector.evaluate(im)
        return im, preds.detach().cpu().numpy(), logits, loss

    def best_scores(self, scores):        
        return scores[1:]

    def filter_detection_masks(self):
        # The detector will usually overestimate the detections and some noisy pixels near the borders
        # will cause trouble with accurate size tracking. 
        # perform a sliding window filtering with an a correspondence process and image difference to denoise the detection regions. 
        # This is a destructive process. The proposal pixel_values and regions are eliminated and a single coalesced and filtered pixel_values and region are generated per frame per object

        for i in range(self.REGION_FILTER_WINDOW_SIZE // 2):
            
            for obj in self.objects:
        
                # copy any necesary information here
                region_maps = {key: obj.get_containing_region(key) for key in obj.regions.keys()}
                pixel_maps = {key: obj.get_containing_pixels(key) for key in obj.frame_pixels.keys()}
                mask_maps = {key: obj.get_containing_mask(key) for key in obj.pred_mask.keys()}

                for key in obj.frame_pixels.keys():
                    
                    keys = range(key - self.REGION_FILTER_WINDOW_SIZE // 2, key + self.REGION_FILTER_WINDOW_SIZE // 2 + 1)
                    regions = list(filter(lambda x: x is not None, [region_maps.get(r, None) for r in keys]))
                    pixel_regions = list(filter(lambda x: x is not None, [pixel_maps.get(r, None) for r in keys]))
                    sizes = [size_from_bbox(r) for r in regions]

                    ref = pixel_maps.get(key)                    
                    ref_flipped = np.flip(ref, axis=(0, 1))
                    mask = mask_maps.get(key)
                    scores_tl = []
                    scores_br = []
                    region = region_maps.get(key)

                    print(f"Pass: {i}, key {key}, region {region}")

                    for k, im in zip(keys, pixel_regions):
                        if k == key:
                            continue
                        # determine the cross correlation for the top-left and bottom-right corners and get the padding for this frame
                        scores_tl.append(self.best_scores(patch_matching_cross_correlation(ref, im)))
                        scores_br.append(self.best_scores(patch_matching_cross_correlation(ref_flipped, np.flip(im, axis=(0, 1)))))
                    
                    # filter scores, padding only, not considering "growing" by using negative values
                    #print(scores_tl)
                    #print(scores_br)
                    offsets_tl = np.array(list(map(lambda x: np.clip(x[1], 0, np.inf), filter(lambda x: x[0] > self.REGION_FILTER_THRESHOLD, scores_tl))))
                    offsets_br = np.array(list(map(lambda x: np.clip(x[1], 0, np.inf), filter(lambda x: x[0] > self.REGION_FILTER_THRESHOLD, scores_br))))
                    
                    #print(offsets_tl)
                    #print(offsets_br)                
                    
                    pad_tl = np.ceil(offsets_tl.mean(axis=0)).astype(np.int32)
                    pad_br = np.ceil(offsets_br.mean(axis=0)).astype(np.int32)

                    #print(pad_tl)
                    #print(pad_br)                
                    
                    if pad_tl.size < 2:
                        pad_tl = [0, 0]
                    if pad_br.size < 2:
                        pad_br = [0, 0]

                    new_region = (region[0] + pad_tl[0], region[1] + pad_tl[1], region[2] - pad_br[0], region[3] - pad_br[1])
                    ref = ref[pad_tl[0]:ref.shape[0]-pad_br[0], pad_tl[1]:ref.shape[1]-pad_br[1]]
                    mask = mask[pad_tl[0]:mask.shape[0]-pad_br[0], pad_tl[1]:mask.shape[1]-pad_br[1]]
                    

                    obj.clear(key)
                    obj.update(key, ref, new_region, mask)

        for obj in self.objects:
            for key in obj.frame_pixels.keys():
                region = obj.get_containing_region(key)
                print(f"Final: key {key}, region {region}")


    def is_region_valid(self, bbox):
        return abs(bbox[0] - bbox[2]) >= self.REGION_MIN_DIMENSION and abs(bbox[1] - bbox[3]) >= self.REGION_MIN_DIMENSION

    def determine_objects_from_masks(self, idx, im, preds):

        labeled = measure.label(preds.astype(np.uint8), connectivity=1)
        regions = measure.regionprops(labeled)
        
        #norm_im = convert_image(im)
        #cv2.imwrite(f"./tracking/patches/frame_{idx}.jpg", norm_im)

        for i, region in enumerate(regions):

            if not self.is_region_valid(region.bbox):
                continue

            # extract region pixels
            s = slice_from_bbox(region.bbox)
            
            bbox = region.bbox
            # scikit bbox are half open, while our bboxes are fully closed, adjust them
            bbox = (bbox[0], bbox[1], bbox[2] - 1, bbox[3] - 1)
            

            region_pixels = im[:, bbox[0]:bbox[2]+1, bbox[1]:bbox[3]+1].mean(axis=0)
            pred_mask = preds[slice_from_bbox(bbox)]
            #assert(region_pixels.shape == size_from_bbox(region.bbox))
            #norm_im = convert_image(region_pixels)
            
            #cv2.imwrite(f"./tracking/patches/frame_{idx}_{i}.jpg", norm_im)            
            #print(region_pixels.shape, region_pixels.dtype)
            
            # compare the region with previously tracked objects
            tracking_object = None
            for obj in self.objects:

                overlaps = obj.overlaps(idx - 1, region.bbox)                
                
                # similarity_score > self.SIMILARITY_THRESHOLD: # This could be used for more complex tracking!
                # similarity_score = obj.compare(idx - 1, region_pixels)[0].max()

                # Other signatures can be looked at for tracking like velocity and direction of travel for matching

                if overlaps > self.IOU_THRESHOLD: 
                    
                    # similar object found
                    tracking_object = obj
                    break            
                    
            if tracking_object == None:
                # new object
                id = len(self.objects)
                print("New object detected: ", region.bbox, id)
                tracking_object = TrackingObject(id)
                self.objects.append(tracking_object)

            tracking_object.update(idx, region_pixels, bbox, pred_mask)

    def log_active_tracks(self, idx):
        print(f"Active tracks for idx {idx}:")
        for obj in self.objects:
            region = obj.regions.get(idx)
            if region is not None:
                print("   ", obj.num_id, region)
            
    def track(self, offset, im):

        # do detection in batch
        im, pred_mask, logits, loss = self.get_detection_masks(im)
        
        for idx, image in enumerate(im):
            mask = pred_mask[idx]
            self.determine_objects_from_masks(offset + idx, image.detach().cpu().numpy(), mask)
            self.log_active_tracks(offset + idx)
        
        return im, pred_mask, logits, loss

    def find_good_keyframes_median(self, obj):

        conv_filter = np.ones(self.SLIDING_WINDOW_SIZE) / self.SLIDING_WINDOW_SIZE

        # determine tracking stability by considering region sizes - stable sizes == good detections
        sizes = obj.get_sizes_as_array() # note: there can be missing frames in this list
        min_frame = min(obj.regions.keys())
        max_frame = max(obj.regions.keys())

        if sizes.shape[0] < self.SLIDING_WINDOW_SIZE:
            # object does not have enough detections to consider
            return None

        # convolve over a window to determine the sliding window mean
        sizes_median = sliding_window_median(sizes.T, (1, self.SLIDING_WINDOW_SIZE))
        #sizes_mean = np.apply_along_axis(lambda x: np.convolve(x, conv_filter, mode="same"), 1, sizes.T)

        # toss outliers and compute the averaged size in the window

        pad_left = (self.SLIDING_WINDOW_SIZE - 1) // 2
        pad_right = self.SLIDING_WINDOW_SIZE - 1 - pad_left

        arr_padded = np.pad(sizes.T, ((0, 0), (pad_left, pad_right)), mode='constant')

        windows = sliding_window_view(arr_padded, (1, self.SLIDING_WINDOW_SIZE))

        # Compute the standard deviation over the last two axes (the window dimensions)
        
        std_map = windows.std(axis=(-2, -1)) * 0.25
        valid_items = np.all(
            (sizes.T >= sizes_median - std_map) & (sizes.T <= sizes_median + std_map),
            axis=0
        )
        keyframes = np.where(valid_items)[0]
        
        sizes_mean_kf = np.round(np.apply_along_axis(lambda x: np.convolve(x, conv_filter, mode="same"), 1, sizes.T[:,keyframes]).T if len(keyframes) > 0 else np.array([]))

        keyframes_nums = list(filter(lambda x: x in obj.regions.keys(), keyframes + min_frame))

        if len(keyframes) == 0:
            # object is too unstable to consider
            return None

        return keyframes_nums, sizes_mean_kf
            
    def find_good_keyframes_interp(self, obj):

        conv_filter = np.ones(self.SLIDING_WINDOW_SIZE)
        
        sizes = obj.get_sizes_as_array() # note: there can be missing frames in this list
        keys = list(obj.frame_pixels.keys())

        min_frame = min(obj.regions.keys())
        max_frame = max(obj.regions.keys())


        if sizes.shape[0] < self.SLIDING_WINDOW_SIZE:
            # object does not have enough detections to consider
            return None

        # calculate the gradient of the sizes
        diff_1st = np.diff(sizes, 1, axis=0)

        # pad by 1 to avoid removing the last frame on differencing
        diff_1st = np.pad(diff_1st, ((0, 1), (0, 0)))

        #diff_2nd = np.diff(sizes, 2, axis=0)
        filtered_diff = np.apply_along_axis(lambda x: np.convolve(x, conv_filter, mode="same"), 1, diff_1st)

        # use the frames where the sizes are stable
        valid_items = np.where(
            (np.abs(filtered_diff[:,0]) < sizes[:,0] * 0.1) & (np.abs(filtered_diff[:,1]) < sizes[:,1] * 0.1)            
        )[0]
        invalid_items = np.where(
            (np.abs(filtered_diff[:,0]) >= sizes[:,0] * 0.1) | (np.abs(filtered_diff[:,1]) >= sizes[:,1] * 0.1)            
        )[0]

        print(valid_items)

        if len(valid_items) == 0:
            # object is too unstable to consider
            return None

        valid_keys = [keys[i] for i in valid_items]
        invalid_keys = [keys[i] for i in invalid_items]
        
        # linearly interpolate
        sizes_valid = sizes[valid_items,:]               
        sizes_filtered = np.zeros((max_frame - min_frame + 1, 2))
        sizes_filtered[np.array(valid_keys) - min_frame] = sizes_valid
        
        # this seems tough to vectorize.... might need to rethink the approach
        if len(invalid_items) > 0:
            for i in range(len(invalid_items)):           
                if len(valid_items) == 1:
                    sizes_filtered[invalid_keys[i] - min_frame] = sizes[valid_items[0]]
                else: 
                    last_valid_idx = bisect_left(valid_items, invalid_items[i])
                    last_valid = valid_items[last_valid_idx-1]
                    if last_valid_idx < len(valid_items):
                        next_valid = valid_items[last_valid_idx]
                    else:
                        # special case if invalid is after all valid frames
                        sizes_filtered[invalid_keys[i] - min_frame] = sizes[-1]
                        continue
                
                    sizes_filtered[invalid_keys[i] - min_frame] = (sizes[next_valid] - sizes[last_valid]) / (next_valid - last_valid) * (invalid_items[i] - last_valid) + sizes[last_valid]              

        return valid_keys, sizes_filtered

    def find_good_keyframes(self, obj):

        return self.find_good_keyframes_interp(obj)

    def find_matching_by_bounding_box(self, idx, bboxes):
        
        iou_scores = []

        for obj in self.objects:
            iou_scores.append([obj.overlaps(idx, box) for box in bboxes])
                        
        # find object correspondences by best IoU scores
        iou_scores = np.array(iou_scores)
        keys = [ obj.num_id for obj in self.objects]
        mapping = {}
        
        remaining_obj_idx = list(range(len(keys)))

        for i in range(len(bboxes)):
            sorted_scores_idx = np.argsort(iou_scores[:, i])
            obj_idx = sorted_scores_idx[remaining_obj_idx][-1]
            mapping[keys[obj_idx]] = i
            try:
                remaining_obj_idx.remove(obj_idx)
            except ValueError as e:
                pass
            
        return mapping

    def determine_tracking_trajectories(self) -> tuple[dict[int, dict[int, tuple]], dict[int, dict[int, tuple]]]:

        tracking_bboxes = {}
        correction_bboxes = {}
        for obj in self.objects:

            min_frame = min(obj.regions.keys())
            max_frame = max(obj.regions.keys())

            bboxes = tracking_bboxes[obj.num_id] = {}
            cor_bboxes = correction_bboxes[obj.num_id] = {}

            results = self.find_good_keyframes(obj)

            if results is None:
                continue

            keyframes_nums, sizes_mean_kf = results
        
            occluded_frames = list(filter(lambda x: x not in obj.regions.keys(), range(min_frame, max_frame + 1)))

            for idx, _ in obj.frame_pixels.items():

                # search for closest keyframe
                keyframe_idx = bisect_left(keyframes_nums, idx)                
                keyframe_prev = keyframes_nums[max(0, keyframe_idx - 1)]
                keyframe_next = keyframes_nums[min(len(keyframes_nums) - 1, keyframe_idx)]
                keyframe = keyframe_prev #if abs(keyframe_prev - idx) <= abs(keyframe_next - idx) else keyframe_next                
                
                bbox = obj.get_containing_region(idx)
                bboxes[idx] = bbox

                if idx != keyframe_next:
                    # find a representative region in the keyframe
                    #bbox = obj.get_containing_region(keyframe)
                    (y_off, x_off) = obj.find_best_fit(keyframe, idx)
                    print(f"key {idx}, ", bbox)
                    print(idx, keyframe, y_off, x_off, sizes_mean_kf[idx - min_frame, 0], sizes_mean_kf[idx - min_frame, 1])
                    # reposition and apply the mean width and height
                    cor_bboxes[idx] = (int(bbox[0] - y_off), int(bbox[1] - x_off), int(bbox[0] - y_off + sizes_mean_kf[idx - min_frame, 0]), int(bbox[1] - x_off + sizes_mean_kf[idx - min_frame, 1]))                                    
                else:
                    cor_bboxes[idx] = bbox
            
            for idx in occluded_frames:

                # no regions to compare, just make a best estimate            
                prev = cor_bboxes.get(idx)
                next = cor_bboxes.get(idx + 1)

                if next is None:
                    next = prev
                if prev is None:
                    prev = next
                if prev is None:
                    # uh-oh
                    assert(False)

                bbox = np.stack([prev, next]).mean(axis=0).astype(np.int32)

                # interpolate the tracking position
                cor_bboxes[idx] = bbox

            # window average the correction boxes to fine tune the results
            conv_filter = np.ones(self.FILTER_SLIDING_WINDOW_SIZE) / self.FILTER_SLIDING_WINDOW_SIZE 
            sorted_keys = sorted(cor_bboxes.keys())
            sorted_bboxes = np.array([ cor_bboxes[k] for k in sorted_keys])
            sorted_bboxes = np.pad(sorted_bboxes, ((self.FILTER_SLIDING_WINDOW_SIZE // 2, self.FILTER_SLIDING_WINDOW_SIZE // 2 + 1), (0, 0)), mode='edge')
            filtered_bboxes = np.apply_along_axis(lambda x: np.convolve(x, conv_filter, mode="same"), 0, sorted_bboxes)[self.FILTER_SLIDING_WINDOW_SIZE//2:-self.FILTER_SLIDING_WINDOW_SIZE//2]
            
            cor_bboxes = { k: boxes for k,boxes in zip(sorted_keys, filtered_bboxes.astype(np.int32)) }
            # avoid averaging errors at the ends
            if bboxes.get(sorted_keys[-1]):
                cor_bboxes[sorted_keys[-1]] = bboxes[sorted_keys[-1]]
            if bboxes.get(sorted_keys[0]):
                cor_bboxes[sorted_keys[0]] = bboxes[sorted_keys[0]]

            correction_bboxes[obj.num_id] = cor_bboxes
                
        return tracking_bboxes, correction_bboxes

def generate_tracks(tracker, loader, on_progress=None):
    offset = 0
    num_batches = len(loader)
    for batch_index, batch in enumerate(loader):
        
        pixel_values = batch["pixel_values"].to(DEVICE)
        pixel_values, pred_mask, logits, loss = tracker.track(offset, pixel_values)

        pred_probs = torch.softmax(logits, dim=1)[:,1]
        
        for i in range(pixel_values.size(0)):
            #visualize the tracking        
            im = torch_to_cv2_image(pixel_values[i].detach())  

            im = overlay_mask_on_image(im, pred_probs[i].detach().cpu().numpy(), color=(255, 0, 255), alpha=0.75)                    
            #im = overlay_mask_on_image(im, pred_probs_next[i][1].detach().cpu().numpy(), color=(0, 0, 255), alpha=0.75)                    
            
            for obj in tracker.objects:
                regions = obj.regions.get(offset + i)
                if regions is None:
                    continue
                for region in regions:                
                    y1, x1, y2, x2 = region                    
                    cv2.rectangle(im, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
                    cv2.putText(im, str(obj.num_id), (x1, y2), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 0))

            im = cv2.resize(im, OUTPUT_SIZE_CV, interpolation=cv2.INTER_CUBIC)


            cv2.imwrite(f"./tracking/regions/frame_{offset + i}.jpg", im)
        
        if on_progress:
            on_progress("Generating detections...",float(batch_index)/num_batches)

        offset += len(pixel_values)

def is_box_visible(tracker, bbox):
    return (tracker.detector.IMAGE_OUTPUT_SIZE[0] > bbox[0]+bbox[2] >= 0 or tracker.detector.IMAGE_OUTPUT_SIZE[0] > bbox[0] >= 0) and \
           (tracker.detector.IMAGE_OUTPUT_SIZE[1] > bbox[1]+bbox[3] >= 0 or tracker.detector.IMAGE_OUTPUT_SIZE[1] > bbox[1] >= 0)

def do_tracking_evaluation(args):

    # create paths
    os.makedirs("./tracking/regions", exist_ok=True)
    os.makedirs("./tracking/tracks", exist_ok=True)

    enable_segmentation = args.enable_segmentation
    enable_smooth_tracking = args.enable_smooth_tracking
    enable_detection_tracking = args.enable_detection_tracking
    on_progress = None
    if "progress_callback" in args:
        on_progress = args.progress_callback
    
    model: SegmentationLightning = SegmentationLightning.load_from_checkpoint(args.checkpoint_path).to(DEVICE)
    model.eval()
    
    tracker = Tracker(model)

    eval_dataset = VehicleSegmentationAugmentedEvaluationDataset(
        validation_dir=validation_images_dir,
        data_dir=labels_dir,
        feature_extractor=feature_extractor,
        num_vehicles=args.num_vehicles
    )
    eval_dataloader = DataLoader(eval_dataset, batch_size=8, shuffle=False, num_workers=4)
    
    if not os.path.exists(args.load) or args.num_vehicles > 1:

        generate_tracks(tracker, eval_dataloader, on_progress)
        
        # write output
        if args.num_vehicles == 1:
            torch.save(tracker.to_json(), args.load)

    else:
        tracker.from_json(torch.load(args.load))

    # as a post processing step, filter all of the detections through the sequence to improve the tracking results
    tracker.filter_detection_masks()
    
    if on_progress:
        on_progress("Generating object tracking trajectories", 1.0)        
    tracking_bboxes, corrections = tracker.determine_tracking_trajectories()
    
    size = OUTPUT_SIZE_CV
    orig_size = tracker.detector.IMAGE_OUTPUT_SIZE
    video_out = cv2.VideoWriter("output_video.mp4",
            cv2.VideoWriter_fourcc(*'mp4v'),
            15, size)

    ious = []
    tracked_ious = []
    
    idx = 0
    for batch_idx, data in enumerate(eval_dataloader):

        images = data["pixel_values"]
        boxes = data["boxes"]
        
        for im, boxes in zip(images, boxes):
        
            # use the lower half of the images with the current frame color values, convert to CV format
            im = im[:,tracker.detector.IMAGE_OUTPUT_SIZE[0]:,:]
            im = torch_to_cv2_image(im.detach(), denormalize=True)
            # grab the boxes for the current frame 
            boxes = boxes[:,0,:]

            bbox_mapping = tracker.find_matching_by_bounding_box(idx, boxes)
            is_visible = [is_box_visible(tracker, bbox) for bbox in boxes]
            
            for obj_idx in tracking_bboxes.keys():
                
                bbox = tracking_bboxes[obj_idx].get(idx, None)
                corr_bbox = corrections[obj_idx].get(idx, None)

                # find the associated bbox for this object from the box mapping
                expected_bbox_idx = bbox_mapping.get(obj_idx)

                obj = tracker.objects[obj_idx]
                
                if bbox is not None:                
                    if enable_segmentation:
                        pred_mask = obj.get_containing_mask(idx).astype(np.uint8)
                        pred_mask = np.pad(pred_mask, ((bbox[0], orig_size[0] - bbox[2] - 1), (bbox[1], orig_size[1] - bbox[3] - 1) ))                                                
                        im = overlay_mask_on_image(im, pred_mask, color=(255, 0, 255), alpha=0.75)                    

                    if enable_detection_tracking:
                        y1, x1, y2, x2 = bbox            
                        cv2.putText(im, str(obj.num_id), (x1, y2), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 0))
                        cv2.rectangle(im, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)            
                

                
                if expected_bbox_idx != None and is_visible[expected_bbox_idx]:
                    if bbox is not None:
                        ious.append(iou(boxes[expected_bbox_idx], bbox))
                    else:
                        ious.append(0)
                
                
                if corr_bbox is not None:                
                    if enable_smooth_tracking:                
                        y1, x1, y2, x2 = corr_bbox                
                        cv2.rectangle(im, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2)            
                
                if expected_bbox_idx != None and is_visible[expected_bbox_idx]:
                    if corr_bbox is not None:
                        tracked_ious.append(iou(boxes[expected_bbox_idx], corr_bbox))
                    else:
                        tracked_ious.append(0)
                

            # rescale the image for output
            rescaled_im = cv2.resize(im, size, interpolation=cv2.INTER_CUBIC)    

            cv2.imwrite(f"./tracking/tracks/frame_{idx}.jpg", rescaled_im)
            video_out.write(rescaled_im)

            if on_progress:
                on_progress("Generating visual outputs...",float(idx)/len(eval_dataset))


            idx += 1

    if video_out:
        video_out.release()

    return {
        "mean_detection_iou": np.mean(ious),
        "mean_tracked_iou": np.mean(tracked_ious),
        "output_video_path": "./output_video.mp4"
    }

    

if __name__ == '__main__':
    
    # load the model
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--checkpoint_path", default="./checkpoints/last.ckpt", required=False)
    argparser.add_argument("--load", default="./regions.pth", required=False)
    argparser.add_argument("--enable_smooth_tracking", default=False, action='store_true')
    argparser.add_argument("--enable_detection_tracking", default=True, action='store_true')
    argparser.add_argument("--enable_segmentation", default=False, action='store_true')
    argparser.add_argument("--num_vehicles", default=1, type=int)
    
    args = argparser.parse_args()

    if not os.path.exists(args.checkpoint_path):
        raise Exception(f"Error: Model path '{args.checkpoint_path}' does not exist")

    metrics = do_tracking_evaluation(args)
    
    print("Mean Detections IoU: ", np.mean(metrics["mean_detection_iou"]))
    print("Mean Tracked IoU: ", np.mean(metrics["mean_tracked_iou"]))
