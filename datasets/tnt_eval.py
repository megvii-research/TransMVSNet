from re import T
from torch.utils.data import Dataset
import numpy as np
import os, cv2
from PIL import Image
from datasets.data_io import *

# Test any dataset with scale and center crop
s_h, s_w = 0, 0
class MVSDataset(Dataset):
    def __init__(self, datapath, listfile, mode, nviews, ndepths=192, interval_scale=1.0,
                max_h=704,max_w=1280, inverse_depth=False, **kwargs):
        super(MVSDataset, self).__init__()
        self.datapath = datapath
        self.mode = mode
        self.nviews = nviews
        self.ndepths = ndepths
        self.interval_scale = interval_scale
        self.fix_res = kwargs.get("fix_res", True)  #whether to fix the resolution of input image.
        self.fix_wh = False
        self.max_h=max_h
        self.max_w=max_w
        self.inverse_depth=inverse_depth
        self.scans = listfile

        self.image_sizes = {'Family': (1920, 1080),
                            'Francis': (1920, 1080),
                            'Horse': (1920, 1080),
                            'Lighthouse': (2048, 1080),
                            'M60': (2048, 1080),
                            'Panther': (2048, 1080),
                            'Playground': (1920, 1080),
                            'Train': (1920, 1080),
                            'Auditorium': (1920, 1080),
                            'Ballroom': (1920, 1080),
                            'Courtroom': (1920, 1080),
                            'Museum': (1920, 1080),
                            'Palace': (1920, 1080),
                            'Temple': (1920, 1080)}

        assert self.mode == "test"
        self.metas = self.build_list()
        print('Data Loader : data_eval_T&T**************' )

    def build_list(self):
        metas = []

        scans = self.scans

        for scan in scans:
            pair_file = "{}/pair.txt".format(scan)
            # read the pair file
            with open(os.path.join(self.datapath, pair_file)) as f:
                num_viewpoint = int(f.readline())
                for view_idx in range(num_viewpoint):
                    ref_view = int(f.readline().rstrip())
                    src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
                    if len(src_views) == 0:
                        continue
                    metas.append((scan, ref_view, src_views))
        print("dataset", self.mode, "metas:", len(metas))
        return metas

    def __len__(self):
        return len(self.metas)

    def read_cam_file(self, filename):
        with open(filename) as f:
            lines = f.readlines()
            lines = [line.rstrip() for line in lines]
        # extrinsics: line [1,5), 4x4 matrix
        extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
        # intrinsics: line [7-10), 3x3 matrix
        intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))
        intrinsics[:2, :] /= 4.0

        depth_min = float(lines[11].split()[0])
        depth_max = float(lines[11].split()[1])
        depth_interval = float((depth_max - depth_min) / self.ndepths)

        return intrinsics, extrinsics, depth_min, depth_interval, depth_max

    def read_img(self, filename):
        img = Image.open(filename)

        np_img = np.array(img, dtype=np.float32) / 255.
        return np_img

    def center_img(self, img): # this is very important for batch normalization
        img = img.astype(np.float32)
        var = np.var(img, axis=(0,1), keepdims=True)
        mean = np.mean(img, axis=(0,1), keepdims=True)
        return (img - mean) / (np.sqrt(var) )

    def read_depth(self, filename):
        # read pfm depth file
        return np.array(read_pfm(filename)[0], dtype=np.float32)

    def scale_mvs_input(self, img, intrinsics, max_w, max_h, base=32):
        h, w = img.shape[:2]
        if h > max_h or w > max_w:
            scale = 1.0 * max_h / h
            if scale * w > max_w:
                scale = 1.0 * max_w / w
            new_w, new_h = scale * w // base * base, scale * h // base * base
        else:
            new_w, new_h = 1.0 * w // base * base, 1.0 * h // base * base

        scale_w = 1.0 * new_w / w
        scale_h = 1.0 * new_h / h
        intrinsics[0, :] *= scale_w
        intrinsics[1, :] *= scale_h

        img = cv2.resize(img, (int(new_w), int(new_h)))

        return img, intrinsics

    def __getitem__(self, idx):
        global s_h, s_w
        meta = self.metas[idx]
        scan, ref_view, src_views = meta
        # img_w, img_h = self.image_sizes[scan]

        if self.nviews>len(src_views):
              self.nviews=len(src_views)+1

        # use only the reference view and first nviews-1 source views
        view_ids = [ref_view] + src_views[:self.nviews - 1]

        imgs = []
        depth_values = None
        proj_matrices = []

        for i, vid in enumerate(view_ids):
            img_filename = os.path.join(self.datapath, '{}/images/{:0>8}.jpg'.format(scan, vid))
            proj_mat_filename = os.path.join(self.datapath, '{}/cams_1/{:0>8}_cam.txt'.format(scan, vid))

            img = (self.read_img(img_filename))
            # imgs.append(self.read_img(img_filename))
            intrinsics, extrinsics, depth_min, depth_interval, depth_max = self.read_cam_file(proj_mat_filename)

            img, intrinsics = self.scale_mvs_input(img, intrinsics,  self.image_sizes[scan][0],  self.image_sizes[scan][1])
            # img, intrinsics = self.scale_mvs_input(img, intrinsics,  self.max_w,  self.max_h)

            if self.fix_res:
                # using the same standard height or width in entire scene.
                s_h, s_w = img.shape[:2]
                self.fix_res = False
                self.fix_wh = True

            if i == 0:
                if not self.fix_wh:
                    # using the same standard height or width in each nviews.
                    s_h, s_w = img.shape[:2]

            c_h, c_w = img.shape[:2]
            if (c_h != s_h) or (c_w != s_w):
                scale_h = 1.0 * s_h / c_h
                scale_w = 1.0 * s_w / c_w
                img = cv2.resize(img, (s_w, s_h))
                intrinsics[0, :] *= scale_w
                intrinsics[1, :] *= scale_h

            imgs.append(img)
            # extrinsics, intrinsics
            proj_mat = np.zeros(shape=(2, 4, 4), dtype=np.float32)  #
            proj_mat[0, :4, :4] = extrinsics
            proj_mat[1, :3, :3] = intrinsics
            proj_matrices.append(proj_mat)

            if i == 0:  # reference view
                if self.inverse_depth is False:
                    depth_values = np.arange(depth_min, depth_interval * (self.ndepths) + depth_min, depth_interval,
                                         dtype=np.float32)
                else:
                    print('********* Here we use inverse depth for all stage! ***********')
                    depth_end = depth_max - depth_interval / self.interval_scale
                    depth_values = np.linspace(1.0 / depth_end, 1.0 / depth_min, self.ndepths, endpoint=False)
                    depth_values = 1.0 / depth_values
                    depth_values = depth_values.astype(np.float32)


        imgs = np.stack(imgs).transpose([0, 3, 1, 2]) # B,C,H,W
        proj_matrices = np.stack(proj_matrices)

        stage2_pjmats = proj_matrices.copy()
        stage2_pjmats[:, 1, :2, :] = proj_matrices[:, 1, :2, :] * 2
        stage3_pjmats = proj_matrices.copy()
        stage3_pjmats[:, 1, :2, :] = proj_matrices[:, 1, :2, :] * 4

        proj_matrices_ms = {
            "stage1": proj_matrices,
            "stage2": stage2_pjmats,
            "stage3": stage3_pjmats
        }

        return {"imgs": imgs,
                "proj_matrices": proj_matrices_ms,
                "depth_values": depth_values,
                "filename": scan + '/{}/' + '{:0>8}'.format(view_ids[0]) + "{}"}