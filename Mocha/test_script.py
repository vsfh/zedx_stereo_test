
import argparse
import torch
parser = argparse.ArgumentParser()
name = 'mocha'

if name == 'mocha':
    from core.mocha_stereo import Mocha
    parser.add_argument('--mixed_precision', default=True, action='store_true', help='use mixed precision')

    # Architecure choices
    parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg", help="correlation volume implementation")
    parser.add_argument('--shared_backbone', action='store_true', help="use a single backbone for the context and feature encoders")
    parser.add_argument('--corr_levels', type=int, default=2, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")
    parser.add_argument('--max_disp', type=int, default=192, help="max disp of geometry encoding volume")

    args = parser.parse_args()
    model = Mocha(args)
    checkpoint = torch.load("/home/feihongshen/code/zedx_stereo_test/Mocha/checkpoints/5_epoch_mocha-stereo.pth.gz", map_location='cpu')


if name== 'igev':
    from core.igev_stereo import IGEVStereo
    parser.add_argument('--restore_ckpt', help="restore checkpoint", default=None)
    parser.add_argument('--dataset', help="dataset for evaluation", default='sceneflow', choices=["eth3d", "kitti", "sceneflow", "middlebury"])
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument("--precision_dtype",default="float16",choices=["float16", "bfloat16", "float32"],help="Choose precision type: float16 or bfloat16 or float32")
    parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during forward pass')

    # Architecure choices
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")
    parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg", help="correlation volume implementation")
    parser.add_argument('--shared_backbone', action='store_true', help="use a single backbone for the context and feature encoders")
    parser.add_argument('--corr_levels', type=int, default=2, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    parser.add_argument('--max_disp', type=int, default=192, help="max disp of geometry encoding volume")
    args = parser.parse_args()

    model = IGEVStereo(args)
    checkpoint = torch.load("/home/feihongshen/code/zedx_stereo_test/Selective-IGEV/checkpoints/sceneflow/8_igev-stereo.pth", map_location='cpu')
    
if name== 'raft':
    from core.raft import RAFT
    parser.add_argument('--restore_ckpt', help="restore checkpoint", default=None)
    parser.add_argument('--dataset', default='sceneflow', help="dataset for evaluation", choices=["eth3d", "kitti", "sceneflow", "middlebury"])
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument("--precision_dtype",default="float16",choices=["float16", "bfloat16", "float32"],help="Choose precision type: float16 or bfloat16 or float32")
    parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during forward pass')
    parser.add_argument('--hidden_dim', type=int, default=128, help="hidden state and context dimensions")
    parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg", help="correlation volume implementation")
    parser.add_argument('--shared_backbone', action='store_true', help="use a single backbone for the context and feature encoders")
    parser.add_argument('--corr_levels', type=int, default=4, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    args = parser.parse_args()

    model = RAFT(args)
    checkpoint = torch.load("/home/feihongshen/code/zedx_stereo_test/Selective-RAFT/checkpoints/sceneflow/ckpt/19_selective-raft.pth", map_location='cpu')

model.load_state_dict({key[7:]: checkpoint[key] for key in checkpoint.keys()}, strict=True)
model.cuda()
model.eval()
    
def test_single(name='mocha'):
    import torch
    import cv2
    import numpy as np
    import struct

    def get_pointcloud(color_image,depth_image,camera_intrinsics):
        """ creates 3D point cloud of rgb images by taking depth information
            input : color image: numpy array[h,w,c], dtype= uint8
                    depth image: numpy array[h,w] values of all channels will be same
            output : camera_points, color_points - both of shape(no. of pixels, 3)
        """

        image_height = depth_image.shape[0]
        image_width = depth_image.shape[1]
        pixel_x,pixel_y = np.meshgrid(np.linspace(0,image_width-1,image_width),
                                    np.linspace(0,image_height-1,image_height))
        camera_points_x = np.multiply(pixel_x-camera_intrinsics[0,2],depth_image/camera_intrinsics[0,0])
        camera_points_y = np.multiply(pixel_y-camera_intrinsics[1,2],depth_image/camera_intrinsics[1,1])
        camera_points_z = depth_image
        camera_points = np.array([camera_points_x,camera_points_y,camera_points_z]).transpose(1,2,0).reshape(-1,3)

        color_points = color_image.reshape(-1,3)

        # Remove invalid 3D points (where depth == 0)
        valid_depth_ind = np.where(depth_image.flatten() > 0)[0]
        camera_points = camera_points[valid_depth_ind,:]
        color_points = color_points[valid_depth_ind,:]

        return camera_points,color_points
    
    def write_pointcloud(filename, xyz_points, rgb_points=None):

        """ creates a .pkl file of the point clouds generated
        """

        assert xyz_points.shape[1] == 3,'Input XYZ points should be Nx3 float array'
        if rgb_points is None:
            rgb_points = np.ones(xyz_points.shape).astype(np.uint8)*255
        assert xyz_points.shape == rgb_points.shape,'Input RGB colors should be Nx3 float array and have same size as input XYZ points'

        # Write header of .ply file
        fid = open(filename,'wb')
        fid.write(bytes('ply\n', 'utf-8'))
        fid.write(bytes('format binary_little_endian 1.0\n', 'utf-8'))
        fid.write(bytes('element vertex %d\n'%xyz_points.shape[0], 'utf-8'))
        fid.write(bytes('property float x\n', 'utf-8'))
        fid.write(bytes('property float y\n', 'utf-8'))
        fid.write(bytes('property float z\n', 'utf-8'))
        fid.write(bytes('property uchar red\n', 'utf-8'))
        fid.write(bytes('property uchar green\n', 'utf-8'))
        fid.write(bytes('property uchar blue\n', 'utf-8'))
        fid.write(bytes('end_header\n', 'utf-8'))
        
        all_points = np.hstack([xyz_points, rgb_points])
        data_buffer = b"".join(
            struct.pack("fffBBB", *(row[:3].tolist() + row[3:].astype(int).tolist())) for row in all_points
        )
        fid.write(data_buffer)
        fid.close()
    
    focal_length = 246.06433333
    baseline = 119.987
    K = np.array([[246.06433333,   0.        , 322.04533333],
        [  0.        , 255.79770667, 204.09618667],
        [  0.        ,   0.        ,   1.        ]])

    img_left_path = '/mnt/ssd4/ZEDX/train_data/20250731/thin14/image_left/left_000051_1753932068412095000.png'
    # img_left_path = '/mnt/ssd4/ZEDX/train_data/20250731/black18/image_left/left_000032_1753958214039995000.png'
    img_right_path = '/mnt/ssd4/ZEDX/train_data/20250731/thin14/image_right/right_000051_1753932068412095000.png'
    # img_right_path = '/mnt/ssd4/ZEDX/train_data/20250731/black18/image_right/right_000032_1753958214039995000.png'

    img_left_ori = cv2.imread(img_left_path)
    img_left_ori = cv2.resize(img_left_ori, (640, 416))
    img_right_ori = cv2.imread(img_right_path)
    img_right_ori = cv2.resize(img_right_ori, (640, 416))

    img_left = torch.from_numpy(img_left_ori.astype(np.float32) / 255.).unsqueeze(0).permute(0,3,1,2).cuda()
    img_right = torch.from_numpy(img_right_ori.astype(np.float32) / 255.).unsqueeze(0).permute(0,3,1,2).cuda()

    with torch.no_grad():
        pred_disp = model(img_left, img_right, iters=32, test_mode=True)
            
    disp = pred_disp[-1].cpu().squeeze().numpy()

    disp[disp < 1] = 1
    disp[disp > 640] = 1
    depth = baseline * focal_length / (disp + 1e-6)
    depth[depth < 0] = 0
    depth[depth > 20000] = 0

    camera_points, color_points = get_pointcloud(img_left_ori, depth.astype(np.uint16), K)
    write_pointcloud(f"depth_{name}.ply", camera_points, color_points)

def test_batch(name='mocha'):
    from dataset_zedx import fetch_loader
    from tqdm import tqdm
    import numpy as np
    import cv2
    test_data_list = "/mnt/ssd4/xingzenglan/libra/data_lists/zedx_val.list"
    train_loader = fetch_loader(test_data_list, bs=1)
    for i_batch, (img_path, *data_blob) in enumerate(tqdm(train_loader)):
        image1, image2, disp_gt, valid = [x.cuda() for x in data_blob]
        with torch.no_grad():
            pred_disp = model(image1, image2, iters=32, test_mode=True)
        disp = pred_disp[-1].squeeze()
        error = torch.abs(disp-disp_gt.squeeze()).cpu().numpy()
        error_map = np.zeros_like(error)
        error_map[error>5] = 255
        error_map[error>3] = 200
        error_map[error>1] = 150
        write_img_path = img_path[0][0].split('/')[-1]
        cv2.imwrite(f'/home/feihongshen/code/zedx_stereo_test/outputs/{name}_{write_img_path}', error_map)
        # break
            
def search_file():
    import glob
    path_list = glob.glob('/mnt/ssd4/ZEDX/train_data/*/*/image_left/left_000051_1753932068412095000.png')
    print(path_list)
    
# test_single('raft')
test_batch()