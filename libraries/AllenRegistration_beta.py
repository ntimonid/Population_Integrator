
from cfg import *
import convertAllenSpace as CAS
import VisuAlignLib as VAL
sys.path.append('atlasoverlay')
# import atlasoverlay as ao
# from BrainPlotter_beta import json_to_flat
from utils import *
from flatmapper import FlatMapper


def get_section_info(in_dir):
    try:
        find_mark_file = glob.glob(os.path.join(in_dir, '*_visualign.json'))
        in_file = find_mark_file[0]
        with open(in_file) as f:
            vafile = json.load(f)
        sorted_section_ids = np.argsort([vafile["slices"][i]['filename'] for i in range(len(vafile["slices"]))])

        section_dict = OrderedDict()
        for i in sorted_section_ids:
            if "markers" not in vafile["slices"][i].keys():
                print("markers field not present: moving to another slice!")
                continue
            section_dict[vafile["slices"][i]['filename']] = vafile["slices"][i]

        return section_dict

    except:
        print('Visualign json file not found. Please provide a valid one for your experiment of interest')
        return -1


class AllenDataStore:

    annotation = None
    acr2id = None
    id2acr = None
    ancestorsById = None,
    template = None
    EUAL_annotation = None

    def __init__(self, flatmapper_dir='../', atlas_dir='../', in_res=10):

        self.flatmapper_dir = flatmapper_dir
        self.atlas_dir = atlas_dir
        self.in_res = in_res

        if AllenDataStore.annotation is None:
            self.annotation, self.acr2id, self.id2acr, \
                self.ancestorsById, self.template, self.EUAL_annotation = getAtlasData(atlas_dir, in_res)
        else:
            self.annotation = AllenDataStore.annotation
            self.template = AllenDataStore.template
            self.EUAL_annotation = AllenDataStore.EUAL_annotation
            self.acr2id = AllenDataStore.acr2id
            self.id2acr = AllenDataStore.id2acr
            self.ancestorsById = AllenDataStore.ancestorsById

        self.Q_neuron2allen = CAS.convertAllenSpace(fromUnitOrientationOrigin=['um(25)', 'RAS', 'corner'],\
                                                    toUnitOrientationOrigin=[f'um({in_res})', 'PIR', 'corner'])
        self.flt_mapper = FlatMapper(self.flatmapper_dir)


class ImageRegistration:

    def __init__(self, AllenDataStore, in_slice, experiment_dir):
        # super().__init__()

        # First, inherit from AllenDataStore instance ...
        self.flt_mapper = AllenDataStore.flt_mapper
        self.Q_neuron2allen = AllenDataStore.Q_neuron2allen
        self.annotation = AllenDataStore.annotation
        self.template = AllenDataStore.template
        self.EUAL_annotation = AllenDataStore.EUAL_annotation
        self.acr2id = AllenDataStore.acr2id
        self.id2acr = AllenDataStore.id2acr
        self.ancestorsById = AllenDataStore.ancestorsById
        self.flatmapper_dir = AllenDataStore.flatmapper_dir
        self.atlas_dir = AllenDataStore.atlas_dir
        self.in_res = AllenDataStore.in_res


        # Additional initialization for image registration if needed
        self.in_slice = in_slice
        self.fname = in_slice["filename"].split('.')[0] if 'import\\' not in in_slice["filename"] else in_slice["filename"].split('import\\')[1].split('.')[0]
        self.width = in_slice["width"]
        self.height = in_slice["height"]
        self.in_slice = in_slice
        self.dims = self.annotation.shape
        self.experiment_dir = experiment_dir


    def done(self):

        self.annotation = None
        self.template = None
        self.Q_neuron2allen = None

    def reverse_transform_pixels(self, def_section_cords):

        # Intialize the transformation triangulation
        triangulation = VAL.triangulate(self.width, self.height, self.in_slice["markers"])

        # Reverse transformation
        section_cords = VAL.forwardtransform_vec(triangulation, def_section_cords[:, 1], def_section_cords[:, 0])

        # !! Warning !! Let's try cleaning the code below. I will put it in comments, try re-writing it and see if it works
        # newCoordsF = np.zeros(section_cords.shape, section_cords.dtype)
        # newCoordsF[:,0] = np.floor(template_cords[1]).astype(np.uint32)
        # newCoordsF[:,1] = np.floor(template_cords[0]).astype(np.uint32)
        #
        # newCoords_clamped = newCoordsF.copy()
        # newCoords_clamped[newCoordsF < 0] = 0
        # newCoords_clamped[newCoordsF[:, 0] >= self.height, 0] = self.height - 1
        # newCoords_clamped[newCoordsF[:, 1] >= self.width, 1] = self.width - 1

        section_cords = np.floor(np.array(section_cords)).astype(np.uint32)
        section_cords[section_cords < 0] = 0
        section_cords[section_cords[:, 0] >= self.height, 0] = self.height - 1
        section_cords[section_cords[:, 1] >= self.width, 1] = self.width - 1

        return section_cords

    def reorder_coordinates(self, section_cords, def_section_cords):

        reorder_list = []
        # section_cords = list(section_cords[0])
        # section_cords = list(section_cords.T)
        section_cords = [list(val) for val in section_cords.T]
        for def_x, def_y in def_section_cords:
            new_index = section_cords.index([def_x, def_y])
            reorder_list.append(new_index)
        reorder_vec = np.asarray(reorder_list)
        section_cords = np.array(section_cords)

        section_cords_reordered = section_cords[reorder_vec]
        def_section_cords_reordered = def_section_cords[reorder_vec]

        return section_cords_reordered, def_section_cords_reordered

    def reverse_transform_image(self, section_cords, def_section_cords):

        # The new transform method
        deformed_img = np.zeros(self.in_img.shape, np.uint8)
        deformed_img[def_section_cords[:,0], def_section_cords[:,1]] = self.in_img[section_cords[:,0], section_cords[:,1]]

        return deformed_img

    def pixel_to_voxel_mapping_original(self):
        # This is the old way used to infer voxels, by first computing the deformed image
        # and then taking its point cloud to make the mapping.

        anchoring = np.array(self.in_slice['anchoring'])
        w, h = self.width, self.height

        o = anchoring[:3]
        u = anchoring[3:6]
        v = anchoring[6:]

        Bp = np.array([u, v, o])
        Bpp = np.array([[1 / w, 0, 0], [0, 1 / h, 0], [0, 0, 1]])

        # Using a third column for ones ..
        self.newimg_F = self.newimg_F.T
        tmp = np.zeros(self.newimg_F.shape,np.uint8)
        newCoordsF_2 = np.argwhere(tmp==0)
        if len(newCoordsF_2[0]) > 2:
            newCoordsF_2[:,2] = 1
        else:
            newCoordsF_2 = np.c_[newCoordsF_2, np.ones(len(newCoordsF_2))]
        newCoordsF_2 = np.array(newCoordsF_2, dtype=int)
        self.newCoordsF_2 = newCoordsF_2

        scaled_cords = Bpp.dot(self.newCoordsF_2.T)

        voxel_cords = scaled_cords.T.dot(Bp)

        voxel_cords_trs = (self.Q_neuron2allen[0:3,0:3] @ voxel_cords.T).T + self.Q_neuron2allen[0:3,3]
        voxel_cords_trs = np.array(np.round(voxel_cords_trs), dtype=int)

        # Voxel filtering
        voxel_cords_trs[:, 0][voxel_cords_trs[:, 0] >= self.dims[0]] = self.dims[0] - 1
        voxel_cords_trs[:, 1][voxel_cords_trs[:, 1] >= self.dims[1]] = self.dims[1] - 1
        voxel_cords_trs[:, 2][voxel_cords_trs[:, 2] >= self.dims[2]] = self.dims[2] - 1

        return voxel_cords_trs

    def pixel_to_voxel_mapping(self):
        # This is the experimental way used to infer voxels, by taking the point cloud
        # of deformed pixel coordinates, which has been reordered to follow the indices
        # of the point cloud of the original pixel coordinates.

        anchoring = np.array(self.in_slice['anchoring'])
        o = anchoring[:3]
        u = anchoring[3:6]
        v = anchoring[6:]

        Bp = np.array([u, v, o])
        Bpp = np.array([[1 / self.width, 0, 0], [0, 1 / self.height, 0], [0, 0, 1]])

        # scaled_cords = Bpp.dot(self.def_section_cords_reordered)
        # self.def_section_cords = self.def_section_cords[:, [1, 0]]

        # def_section_cords_trsp = np.argwhere(self.def_img.T == 0)
        # def_section_cords_ext = np.c_[def_section_cords_trsp, np.ones(len(def_section_cords_trsp))]
        # def_section_cords_ext = np.c_[self.def_section_cords, np.ones(len(self.def_section_cords))]

        def_section_cords_ext = np.c_[self.def_section_cords[:,[1,0]], np.ones(len(self.def_section_cords))]
        def_section_cords_ext = np.array(def_section_cords_ext, dtype=int)

        scaled_cords = Bpp.dot(def_section_cords_ext.T)
        voxel_cords = scaled_cords.T.dot(Bp)

        voxel_cords_trs = (self.Q_neuron2allen[0:3, 0:3] @ voxel_cords.T).T + self.Q_neuron2allen[0:3, 3]
        voxel_cords_trs = np.array(np.round(voxel_cords_trs), dtype=int)

        # Voxel filtering
        voxel_cords_trs[:, 0][voxel_cords_trs[:, 0] >= self.dims[0]] = self.dims[0] - 1
        voxel_cords_trs[:, 1][voxel_cords_trs[:, 1] >= self.dims[1]] = self.dims[1] - 1
        voxel_cords_trs[:, 2][voxel_cords_trs[:, 2] >= self.dims[2]] = self.dims[2] - 1

        return voxel_cords_trs

    def load_image(self):

        path_to_images = f'{self.experiment_dir}/import'
        try:
            in_img = np.array(Image.open('{}/{}'.format(path_to_images, self.fname)))
        except:
            print('Error! This image is not available! Please provide an available image')
            in_img = - 1

        return in_img

    def register_section(self):

        # That's the og way of loading the og image, but I will first attempt registering without loading it
        # self.in_img = self.load_image()
        self.in_img = np.zeros((self.height, self.width), np.uint8)
        self.def_img = np.zeros((self.height, self.width), np.uint8)
        if self.in_img.any() == -1 or self.in_img.shape != self.def_img.shape:
            return -1

        # Reverse transformation
        def_section_cords = np.argwhere(self.def_img == 0)
        section_cords = self.reverse_transform_pixels(def_section_cords)
        self.def_section_cords, self.section_cords = def_section_cords, section_cords
        # self.def_img = reverse_transform_image(section_cords, def_section_cords)
        # self.section_cords_reordered, self.def_section_cords_reordered = \
        #     self.reorder_coordinates(section_cords, def_section_cords)
        print('non-linear reverse transform completed')

        # Voxel mapping
        self.voxel_cords_trs = self.pixel_to_voxel_mapping()
        print(f'2D registration of experiment {self.fname}  has been completed')

        return -1

    def give_me_transformed_cords(self, pixel_cord):

        try:
            return self.newCoordsF_2.index(pixel_cord)
        except:
            print('The requested transformed pixel coordinate is out of bounds. Please select another pixel coordinate')
            return -1

    def give_me_corresponding_voxels(self, pixel_cord):

        try:
            vox_idx = self.section_cords.index(pixel_cord)
            return self.Voxel_Coo_trs[vox_idx]
        except:
            print('The requested voxel coordinate is out of bounds. Please select another pixel coordinate')
            return -1

    def give_me_flatmap_points(self, pixel_cord, projectionType='dorsal_flatmap'):

        self.flt_mapper.init(projectionType)

        try:
            point_cloud_voxel_cord = self.give_me_corresponding_voxels(pixel_cord)
            point_cloud_flat_cord = np.array(self.flt_mapper.projectPoints([point_cloud_voxel_cord]))
            return point_cloud_flat_cord
        except:
            print('Warning! The flatmap could not be computed')
            return -1

    def store_densities(self, dims):

        a = self.Voxel_Coo_trs[:,0]>=dims[0]; b = self.Voxel_Coo_trs[:,1]>=dims[1]; c = self.Voxel_Coo_trs[:,2]>=dims[2]
        flag_pts = np.logical_or(a,np.logical_or(b,c))

        Voxel_Coo_trs_in_border = self.Voxel_Coo_trs[flag_pts==False,:]
        newCoordsF_in_border = self.newCoordsF_2[flag_pts==False,:]

        axon_mask = self.newimg_F[newCoordsF_in_border[:,0], newCoordsF_in_border[:,1]] == 255 # axon num
        axon_voxels = Voxel_Coo_trs_in_border[axon_mask,:]

        for voxel_cord in axon_voxels:
            v1,v2,v3 = voxel_cord
            if self.annotation[v1,v2,v3] == self.acr2id['[background]']: continue
            if tuple(voxel_cord) not in self.target_dict.keys():
                self.target_dict[tuple(voxel_cord)] = 1
            else:
                self.target_dict[tuple(voxel_cord)] += 1


#******
