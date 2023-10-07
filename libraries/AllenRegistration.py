# %%
# import os, sys
# import json, random
# import io
# import base64
# import tempfile
# from PIL import Image
# from lxml import etree
# from copy import deepcopy
# import nrrd,PIL.Image,re,IPython.display

# current_path = os.getcwd()
# main_path = os.path.abspath('../') #'/cortexdisk/data2/NestorRembrandtCollab/'
# # lib_dir = os.path.join(main_path,'Libraries')
# code_dir = os.path.join(main_path,'Code Repositories')
# # sys.path.append(lib_dir)
# # sys.path.append(code_dir)
# sys.path.append(os.path.join(code_dir,'atlasoverlay'))


from cfg import *
import convertAllenSpace as CAS
import VisuAlignLib as VAL
sys.path.append('atlasoverlay')
import atlasoverlay as ao
from BrainPlotter_beta import json_to_flat


def setup(experiment_id, dir_data):
    file_path = f'{dir_data}/{experiment_id}'
    imgdir = f'{file_path}/original_images'
    downdir = f'{file_path}/import'
    required_folders = [imgdir, downdir, f'{file_path}/export']

    for folder in required_folders:
        try:
            os.makedirs(folder)
            print('Folders created for', experiment_id)
        except: pass

    return (file_path, imgdir, downdir)


def addnumbers(imgdir):
    '''
    Simple function to add numbers to the section images, in case they don't already have them.
    This assumes that you add '_cropped' to the input images at the end.
    '''
    imgno = 1
    for file in os.listdir(imgdir):
        if 'cropped' not in file.split('_')[-1]:
            print('Number already added')
            continue
        filename = file.split('.')[0]
        extension = file.split('.')[-1]
        new_filename = f'{filename}_s{str(imgno).zfill(3)}.{extension}'
        imgno += 1
        print(f'{imgdir}/{file}', f'{imgdir}/{new_filename}')
        os.rename(f'{imgdir}/{file}', f'{imgdir}/{new_filename}')


def downsample(imgdir, downdir):
    '''
    Transform images to PNGs and downsample them so that they can be handled by QuickNii.
    '''

    # downsampling image data to something that can be handled by QuickNii (<16 MPixels)

    Mp_const = 10e5 # Million pixels constant (10^6)

    n = len(os.listdir(imgdir))
    for file in os.listdir(imgdir):
        jump = 0
        with Image.open(os.path.join(imgdir, file)) as im:
            # print(os.path.join(imgdir, file))
            width = im.width; height = im.height

            factor = 16 # original was 16
            ds_limit = int(np.ceil(np.sqrt(width*height/(factor*Mp_const))))

            if ds_limit > jump:
                jump = ds_limit

    print('Grabbing 1 pixel every', jump)

    for file in os.listdir(imgdir):
        fname = '_'.join((file.split('_')[0:-1]))+f'_downsampled{jump}_'+file.split('_')[-1][:-4]+'.png'
        if fname in os.listdir(downdir):
            n -=1
            print(f'{file} already downsampled; {n} images remaining. ')
            continue
        with Image.open(os.path.join(imgdir, file)) as im:
            Im_Mat = np.array(im) #'./'
            Im_Mat_ds = Im_Mat[::ds_limit,::ds_limit,:]
            print('Previous pixel area:', width*height/Mp_const)
            print('New pixel area:', Im_Mat_ds.shape[0]*Im_Mat_ds.shape[1]/Mp_const)

            im_ds = Image.fromarray(Im_Mat_ds)
            im_ds.save(os.path.join(downdir, fname))
            n -=1
            print('{} images remaining.'.format(n))

    return ds_limit


def rotate_imgs(downdir, imgs2rot, angle=180, del_oldfile=False):
    exp = downdir.split('/')[-2]
    if exp not in imgs2rot: return(0)
    for file in os.listdir(downdir):
        if '.png' not in file: continue
        if 'rot' in file: continue
        if int(file.split('_')[2]) in imgs2rot[exp]:
            outfilename = '_'.join(file.split('_')[:-1]+[f'rot{angle}']+[file.split('_')[-1]])
            if outfilename in os.listdir(downdir): continue
            im = Image.open(os.path.join(downdir, file))
            im_rot = im.rotate(angle)
            im_rot.save(os.path.join(downdir, outfilename))
            print(file, 'rotated')
            if del_oldfile: os.remove(os.path.join(downdir, file))


def mirror_imgs(downdir, imgs2mir, del_oldfile=False):
    exp = downdir.split('/')[-2]
    if exp not in imgs2mir: return(0)
    for file in os.listdir(downdir):
        if '.png' not in file: continue
        if 'mirror' in file: continue
        if int(file.split('_')[2]) in imgs2mir[exp]:
            outfilename = '_'.join(file.split('_')[:-1]+['mirror']+[file.split('_')[-1]])
            if outfilename in os.listdir(downdir): continue
            im = Image.open(os.path.join(downdir, file))
            im_mir = ImageOps.mirror(im)
            im_mir.save(os.path.join(downdir, outfilename))
            print(file, 'mirrored')
            if del_oldfile: os.remove(os.path.join(downdir, file))

# First major change ... Name of the function
def deepslice_analysis(file_path):
    '''
    Make sure to give the path above the file containing the images (i.e., if your images are in 'path/images', give 'path')
    The function assumes that the folder's name is the name of the experiment, and will use that name for the result file
    '''

    from DeepSlice import DSModel
    # expdir = file_path
    n = file_path.split('/')[-1]
    output = f'{file_path}/{n}_deepslice'

    species = 'mouse' #available species are 'mouse' and 'rat'
    Model = DSModel(species)

    Model.predict(f"{file_path}/import", ensemble=True, section_numbers=True)

    # Model.propagate_angles()
    # Model.enforce_index_spacing()
    Model.save_predictions(output)

    """
    from DeepSlice.DeepSlice import DeepSlice

    n = file_path.split('/')[-1]
    output = f'{file_path}/{n}_deepslice'

    Model = DeepSlice()
    Model.Build()
    Model.predict(file_path) #prop_angles=False
    Model.reorder_indexes(ascending = True)
    Model.even_spacing()
    Model.Save_Results(output)
    """


def xmltojson(xml_file):
    file = xml_file.split('/')[-1]
    path = '/'.join(xml_file.split('/')[:-1])

    tree = ET.parse(xml_file)
    root = tree.getroot()

    slices = []
    for section in root:
        section_dict = {}
        attributes = ['height', 'width', 'filename', 'anchoring', 'nr']
        for attribute in attributes:
            if attribute == 'filename':
                section_dict[attribute] = section.attrib[attribute]
            if attribute in ['height', 'width', 'nr']:
                # Second small change ...
                section_dict[attribute] = int(float(section.attrib[attribute])) #int(section.attrib[attribute])
            if attribute == 'anchoring':
                anchoring = list(float(x) for x in filter(None, re.split('&|\w{2}=', section.attrib['anchoring'])))
                section_dict[attribute] = anchoring
        slices.append(section_dict)

    data = {}
    data['name'] = root.attrib['name']
    data['target'] = 'ABA_Mouse_CCFv3_2015_25um.cutlas'
    data['target-resolution'] = [456,528,320]
    data['slices'] = slices
    outfile = os.path.join(path, file[:-3])+'json'
    with open(outfile, 'w') as f:
        json.dump(data, f)

    return data


def reveal(file_path):
    path_to_images = f'{file_path}/import'
    for file in os.listdir(path_to_images):
        if '.png' and 'SimpleSegmentation' not in file: continue
        img = Image.open(f'{path_to_images}/{file}') # load the image
        #outfile = file.split('.')[0] + 'revealed' + '.png'
        pixel_map = img.load() # create the pixel map from the images
        width, height = img.size # grab dimensions
        for i in range(width):    # for every col:
            for j in range(height):    # for every row
                if pixel_map[i,j] == 1: pixel_map[i,j] = (255) # signal/axon is white
                if pixel_map[i,j] == 2: pixel_map[i,j] = (0) # background is black
                if pixel_map[i,j] == 3: pixel_map[i,j] = (129) # soma is gray
        img = img.save(f'{path_to_images}/{file}')
#         img = img.save(os.path.join(path_to_images,file.split('.')[0] + 'revealed.png'))

def invert(file_path):                           #changes the contrast of the images
    path_to_images = f'{file_path}/import'
    for file in os.listdir(path_to_images):
        if '.png' and 'SimpleSegmentation' not in file: continue
        img = Image.open(f'{path_to_images}/{file}') # load the image
        pixel_map = img.load() # create the pixel map from the images
        width, height = img.size # grab dimensions
        for i in range(width):    # for every col:
            for j in range(height):    # for every row
                if pixel_map[i,j] == 255: pixel_map[i,j] = (0) # background is black
                if pixel_map[i,j] == 0: pixel_map[i,j] = (255) # signal is white
        img = img.save(f'{path_to_images}/{file}')


# def remove_false_somata(InputMat, annotation, acr2id, soma_acr = None, experiment_id = 'None'):
#     # Here I will remove all points marked as somata, that do not correspond to the brain area they should ...
#     if experiment_id == 'None' and soma_acr is None:
#         print('please provide the experiment filename or a ground truth')
#         return -1
#     if len(experiment_id.split('_')) > 2:
#         soma_acr = experiment_id.split('_')[1] # Here's where the soma area should be
#     area_mask = np.array(annotation != acr2id[soma_acr])
#     soma_mask = np.array(InputMat == 129)
#     join_mask = area_mask&soma_mask
#     InputMat[join_mask] = 255
#     return InputMat


def run_pylastik(file_path, classifier_name, ilastik_path, mode):

    path_to_images = f'{file_path}/import'
    if os.path.isdir(path_to_images) is False:
        return -1

    if mode == '"Simple Segmentation"': fmode = 'SimpleSegmentation'
    elif mode == '"Probabilities"': fmode = 'Probabilities'

    for infile_name in os.listdir(path_to_images):
        if 'SimpleSegmentation' in infile_name or 'Probabilities' in infile_name or '.png' not in infile_name: continue
        fname = infile_name.split('.png')[0]
        outfile_name = '{}_{}.png'.format(fname, fmode)

        current_path = os.getcwd()
        os.chdir(ilastik_path)

        # macOS:
        if sys.platform == 'OS':
            get_ipython().system('./run_ilastik.sh --export_source=$mode --project=$current_path/Ilastik_Classifiers/$classifier  --output_format=png  --output_filename_format=$file_path/$file_folder/$outfile_name  $file_path/import/$infile_name')

        # Windows: counting steps to D: from parent directory could fix hardcoding?
        elif sys.platform == 'windows':
            get_ipython().system(' .\\ilastik.exe --headless --export_source=$mode --project=$current_path/Ilastik_Classifiers/$classifier  --output_format=png  --output_filename_format=.\\$file_path\\$file_folder\\$outfile_name  $file_path/import/$infile_name')

        # Linux:
        elif sys.platform == 'linux':
            get_ipython().system('./run_ilastik.sh --headless --export_source=$mode --project=$current_path/Ilastik_Classifiers/$classifier_name  --output_format=png  --output_filename_format=$file_path/import/$outfile_name  $file_path/import/$infile_name')

        os.chdir(current_path)

        # let's do the reveal here instead ...
        img = Image.open(f'{path_to_images}/{outfile_name}') # load the image
        #outfile = file.split('.')[0] + 'revealed' + '.png'
        pixel_map = img.load() # create the pixel map from the images
        width, height = img.size # grab dimensions
        for i in range(width):    # for every col:
            for j in range(height):    # for every row
                if pixel_map[i,j] == 1: pixel_map[i,j] = (255) # signal/axon is white
                if pixel_map[i,j] == 2: pixel_map[i,j] = (0) # background is black
                if pixel_map[i,j] == 3: pixel_map[i,j] = (129) # soma is gray
        img = img.save(f'{path_to_images}/{outfile_name}')

# That's the old run_pylastik I keep it because we may need to revert to it
# def run_pylastik(file_path, file_folder, outfile_name, infile_name, classifier, ilastik_path = './', mode = '"Simple Segmentation"', out_format='png'):
#     # file_path: path leading up to the nested directory where your images-for-segmentation are
#     # file_folder: name of the folder where your input images are
#     # outfile_name: name of the output segmented file:
#     # infile_name: name of the actual file for segmentation:
#     # classifier: name of your classifier .ilp file
#     # ilastik_path: path leading to the directory of the classifier. Default: './'
#     # mode: segmentation mode: can be either Simple Segmentation or Probabilities. Default: "Simple Segmentation"
#     # out_format: output file format. Default: "png"
#
#     current_path = os.getcwd()
#     os.chdir(ilastik_path)
#
#     # macOS:
#     if sys.platform == 'OS':
#         get_ipython().system('./run_ilastik.sh --export_source=$mode                           --project=Ilastik_Classifiers/$classifier                           --output_format=png                           --output_filename_format=$file_path/$file_folder/$outfile_name                            $file_path/$file_folder/$infile_name')
#
#     # Windows: counting steps to D: from parent directory could fix hardcoding?
#     elif sys.platform == 'windows':
#         get_ipython().system(' .\\ilastik.exe --headless                        --export_source=$mode                        --project=Ilastik_Classifiers/$classifier                        --output_format=png                        --output_filename_format=.\\$file_path\\$file_folder\\$outfile_name                         $file_path/$file_folder/$infile_name')
#
#     # Linux:
#     elif sys.platform == 'linux':
#         get_ipython().system('./run_ilastik.sh --headless                        --export_source=$mode                        --project=$current_path/Ilastik_Classifiers/$classifier                        --output_format=png                        --output_filename_format=$file_path/$file_folder/$outfile_name                        $file_path/$file_folder/$infile_name')
#
#     os.chdir(current_path)

# old pylastik code, check new
# def run_pylastik(dir1, dir2, out, elem, classifier, ilastik_path = './', mode = '"Simple Segmentation"', out_format='png'):
#     '''
#     Windows only. And even on Windows it can fail, depending on your folder structure.
#     '''
#     current_path = os.getcwd()
#     os.chdir(ilastik_path)

#     # macOS:
#     if sys.platform == 'OS':
#         get_ipython().system('./run_ilastik.sh --export_source=$mode                           --project=Ilastik_Classifiers/$classifier                           --output_format=png                           --output_filename_format=$dir1/$dir2/$out                            $dir1/$dir2/$elem')

#     # Windows: counting steps to D: from parent directory could fix hardcoding?
#     elif sys.platform == 'windows':
#         get_ipython().system(' .\\ilastik.exe --headless                        --export_source=$mode                        --project=Ilastik_Classifiers/$classifier                        --output_format=png                        --output_filename_format=.\\$dir1\\$dir2\\$out                         $dir1/$dir2/$elem')

#     # Linux:
#     elif sys.platform == 'linux':
#         get_ipython().system('./run_ilastik.sh --headless                        --export_source=$mode                        --project=$current_path/Ilastik_Classifiers/$classifier                        --output_format=png                        --output_filename_format=$dir1/$dir2/$out                        $dir1/$dir2/$elem')

#     os.chdir(current_path)


def sortchildrenby(parent, attr):
    parent[:] = sorted(parent, key=lambda child: child.get(attr))



# def split_hemisphere(InputMat):

    # ## reference  points for the splitting
    # last_point = np.array(InputMat.shape)
    # mid_point = np.round(last_point/2).astype(int) # middle point of the right-left axis
    #
    # ## MatrixL keeps the left hemisphere projections while MatrixR keeps the right hemisphere one
    # if np.issubdtype(type(InputMat[0,0,0]), int) is True:   # check if we are splitting an integer array or a float one
    #     MatrixL = np.zeros(np.shape(InputMat), dtype = np.uint8)
    #     MatrixR = np.zeros(np.shape(InputMat), dtype = np.uint8)
    # else:
    #     MatrixL = np.zeros(np.shape(InputMat), dtype = np.float16)
    #     MatrixR = np.zeros(np.shape(InputMat), dtype = np.float16)
    #
    # MatrixL[:,:,0:mid_point[2]-1] = InputMat[:,:,0:mid_point[2]-1]
    # MatrixR[:,:,0:mid_point[2]-1] = InputMat[:,:,mid_point[2]+1:last_point[2]][:,:,::-1]

# %%

# def clean_segmentation(path, file, min_size=5, connectivity=2):
#
#     try:
#         file = file.split('import\\')[1].split('.')[0]+'_SimpleSegmentation.png'
#     except:
#         file = file.split('.')[0]+'_SimpleSegmentation.png'
#     file = '/import/' + file
#
#     image = cv2.imread(path + '/'+ file)
#
#     blackLower = np.array([129], dtype = "uint8")  # remember there are three categories; was it 120 something for the other?
#     blackUpper = np.array([255], dtype = "uint8")
#
#     black = cv2.inRange(image, blackLower, blackUpper) # thresholding
#
#     imglab = morphology.label(black)
#
#     # remove small objects
#     cleaned = morphology.remove_small_objects(imglab, min_size=min_size, connectivity=connectivity)
#
#     # save the image with the removed values
#     output = np.zeros((cleaned.shape)) # create array of size cleaned
#     output[cleaned > 0] = 255
#     output = np.uint8(output)
#
#     # readding the population (value == 129) to the image
#     popvalues = np.where(image == 129)
#     for n in range(len(popvalues[0])):
#         output[popvalues[0][n], popvalues[1][n]] = 129
#
#     fname = file.split('import/')[1].split('.')[0]
#     cv2.imwrite('/{}/import/{}'.format(path,fname + "_cleaned.png"), output)
#
#     return imglab

def clean_segmentation(file_path, min_size = 12, connectivity = 3): # original size and connectivity were 5 and 2

    path_to_images = f'{file_path}/import'
    for file in os.listdir(path_to_images):
        if '.png' and 'SimpleSegmentation' not in file: continue #if '.png' and 'SimpleSegmentation_revealed' not in file: continue
        print(path_to_images,file)
        image = cv2.imread(path_to_images + '/'+ file)

        blackLower = np.array([129], dtype = "uint8")  # remember there are three categories; was it 120 something for the other?
        blackUpper = np.array([255], dtype = "uint8")

        black = cv2.inRange(image, blackLower, blackUpper) # thresholding

        imglab = morphology.label(black)

        # remove small objects
        cleaned = morphology.remove_small_objects(imglab, min_size=min_size, connectivity=connectivity)

        # save the image with the removed values
        output = np.zeros((cleaned.shape)) # create array of size cleaned
        output[cleaned > 0] = 255
        output = np.uint8(output)

        # readding the population (value == 129) to the image
        popvalues = np.where(image == 129)
        for n in range(len(popvalues[0])):
            output[popvalues[0][n], popvalues[1][n]] = 129

        output_fname = file.split('.')[0] + "_cleaned.png"
        cv2.imwrite('/{}/{}'.format(path_to_images, output_fname), output)

        return imglab

def clean_array(arr):
    arr_clean = np.copy(arr) # copy to return later, after correction
    allvalues = np.unique(arr) # how many different segmentation values are there?
    if len(allvalues) == 3: # the injection value should be in the middle; this only works for 3 values
        back_val = allvalues[0] # background; should always be 0
        inj_val = allvalues[1] # injection; should be 129
        axon_val = allvalues[2] # axons; should be 255
    else:
        print(f"{len(allvalues)} different values found in array. There should be 3 (background, injection, neurites).")
        return -1

    inj_indx = np.where(arr == inj_val) # get the position of all injection values

    for n in range(len(inj_indx[0])):
        vxl_id = annotation[inj_indx[0][n], inj_indx[1][n], inj_indx[2][n]]
        if vxl_id == 0: continue # avoid error due to custom tag introduced by Rembrandt ('[background]')
        if acr2id['TH'] not in ancestorsById[str(vxl_id)]: # if it's not part of the thalamus, change injection to axon
            arr_clean[inj_indx[0][n], inj_indx[1][n], inj_indx[2][n]] = axon_val

    return arr_clean


# def CleanFlatmap(flt_Mat, min_neighbors = 3, radius = 1):

#     # Flatmap before - make a tuple to use for the analysis
#     #tuple_pts = CAS.fast_volume2pts(flt_Mat)
#     nzero_pts = np.nonzero(flt_Mat)
#     tuple_pts = np.array([(x,y) for x,y in zip(nzero_pts[0],nzero_pts[1])])

#     tree = spatial.KDTree(np.array(tuple_pts))
#     neighbors = tree.query_ball_tree(tree, radius)
#     filter_pts = [tuple_pts[idx] for idx,val in enumerate(neighbors) if len(val) >= min_neighbors]
#     print(len(filter_pts),len(tuple_pts))

#     # Flatmap after - make it again a 2D array
#     new_flt_Mat = np.zeros(np.shape(flt_Mat),dtype = np.uint8)
#     for pts in filter_pts:
#         new_flt_Mat[pts[0],pts[1]] = 1

#     return new_flt_Mat

def filter_by_density(arrays, chunk_size=2000, radius=12.0, threshold=0.35):
    points = arrays[0]
    print("Filtering by density")
    points = points.astype(np.uint8)
    n = len(points)
    num_neighbors = []
    for i in range(0, n, chunk_size):
        # print("Processing new chunk")
        chunk_end = min(i + chunk_size, n)
        chunk_points = points[i:chunk_end]
        chunk_distances = distance.cdist(chunk_points, points, "euclidean")
        bool_dist = chunk_distances < radius
        chunk_neighbors = list(np.sum(bool_dist, axis=1))
        num_neighbors.extend(chunk_neighbors)
    num_neighbors = np.array(num_neighbors)
    print(len(num_neighbors))
    # points = points[:20000]    # print(f"Number of pts: {points.shape[0]}")
    # dist_mat = distance.cdist(points, points, "euclidean")
    max_neighbors = np.max(num_neighbors)
    densities = num_neighbors / max_neighbors
    print(np.sum(densities > threshold))
    return [array[densities > threshold] for array in arrays]

def filter_by_area(arrays, areas, resolution=10):
    points = arrays[0]
    print("Filtering by area")
    print(len(points))
    area_filters = []
    total_filter = np.ones((len(points),), dtype=bool)
    for area in areas:
        area_filters.append(area.contains(points * resolution))
    total_filter = np.logical_or.reduce(area_filters)
    print(np.sum(total_filter))
    return [array[total_filter] for array in arrays]

def clean_target_dictionary(voxel_dict,threshold = 0.20):

    voxel_indices = (
        np.vstack([np.array(coord) for coord in voxel_dict.keys()])
    ).astype(float)

    voxel_densities = np.array([coord for coord in voxel_dict.values()])

#     voxel_indices, voxel_densities = filter_by_area([voxel_indices, voxel_densities], cortical_areas) # Not sure about that one yet ...
    voxel_indices, voxel_densities = filter_by_density([voxel_indices, voxel_densities], threshold=threshold)

    return {tuple(i.astype(int)): d for i,d in zip(voxel_indices,voxel_densities)}


# %%
def get_layer_from_area(area):

    layer = [(idx,val) for idx,val in enumerate(area) if val.isnumeric()]
    if len(layer) > 1:
        layer = 'L' + str(layer[0][1] ) + '/' + str(layer[1][1] )
    else:
        if layer[0][0] < len(area) - 1: # there is an a or b there
            layer = 'L' + layer[0][1] + area[len(area) - 1]
        else:
            layer = 'L' + str(layer[0][1])

    return layer

# %%
def xml_to_flat(path, file, atlas, outdir):
    o,u,v = np.array(qn_points(path,file)) * 2.5 # get the data from the xml
    w = int(np.linalg.norm(u)) + 1 # get the 2-norm of vector u, which would be the x coord
    h = int(np.linalg.norm(v)) + 1 # get the 2-norm of vector v (euclidean distance), the y coord
    print(file,w,h)
    # image = PIL.Image.new("RGB",(w,h))
    image = Image.new("RGB",(w,h))
    for y in range(h): # loop through the norm of u
        for x in range(w): # loop through the norm of v
            i,j,k = o + u*x/w + v*y/h # get the coordinates for the plane
            # multiplying each vector by the fraction of its norm and adding the origin value
            # then unpacking the array to get each individual coordinate,
            if 0 <= i < 1140 and 0 <= j < 1320 and 0 <= k < 800: # if within the confines of the array
                image.putpixel((x,y),palette[atlas[0][1319-int(j)][799-int(k)][1139-int(i)]]) # get the rgb for an id
    image.save(f"{outdir}/{file}.png","PNG")
    IPython.display.display(image) # plot it!


class Allen_Registration:

    in_img = None
    def __init__(self, annotation = None, template = None, acr2id = None, soma_acr = None, in_color = '#000',
                 in_directory = '../', in_res = 25, Vox_Mat = None, annot2 = None, data_dir = None,
                 experiment_id = None):

        self.in_res  = in_res
        self.Vox_Mat = Vox_Mat
        self.annotation = annotation
        self.template = template
        self.in_color = in_color
        self.data_dir = data_dir
        self.experiment_id = experiment_id

        self.Q_neuron2allen = CAS.convertAllenSpace(fromUnitOrientationOrigin = ['um(25)','RAS','corner'],\
                                toUnitOrientationOrigin = ['um({})'.format(self.in_res),'PIR','corner'])

        self.acr2id = acr2id
        self.soma_acr = soma_acr
        if template is None:
            self.template,b = nrrd.read(os.path.join(in_directory,'average_template_{}.nrrd'.format(in_res)))
        if annotation is None:
            self.annotation,b = nrrd.read(os.path.join(in_directory,'annotation_{}.nrrd'.format(in_res)))
        if annot2 is not None:
            self.annotation2 = annot2
        else:
            self.annotation2 = None
        if Vox_Mat is None:
            self.Vox_Mat = np.zeros(self.annotation.shape, dtype = np.uint8)

        self.source_dict = {}  # New addition (17/01/2023)
        self.target_dict = {}
        self.voxel_to_pixel_dict = {}

    def done(self):

        self.Vox_Mat = None
        self.annotation = None
        self.annotation2  = None
        self.template = None
        self.Q_neuron2allen = None
        self.voxel_to_pixel_dict = None

    def rev_transform_pixels(self, allCoords, yx):

        self.newCoordsF = np.zeros(allCoords.shape,allCoords.dtype)
        self.newCoordsF[:,0] = np.floor(yx[1]).astype(np.uint32)
        self.newCoordsF[:,1] = np.floor(yx[0]).astype(np.uint32)

        newCoords_clamped = self.newCoordsF.copy()
        newCoords_clamped[self.newCoordsF<0] = 0
        newCoords_clamped[self.newCoordsF[:,0]>=self.in_img.shape[0],0] = self.in_img.shape[0]-1
        newCoords_clamped[self.newCoordsF[:,1]>=self.in_img.shape[1],1] = self.in_img.shape[1]-1

        newimg_tmp = deepcopy(self.newimg_F)
        newimg_tmp[allCoords[:,0],allCoords[:,1]] = self.in_img[newCoords_clamped[:,0],newCoords_clamped[:,1]]

        # The new transform method
        self.newimg_F[allCoords[:,0],allCoords[:,1]] = self.in_img[newCoords_clamped[:,0],newCoords_clamped[:,1]]
        self.newCoordsF = newCoords_clamped
        #plt.imshow(self.newimg_F)

        #return newimg,newCoords_clamped

    def Pixel2Voxel(self, in_slice, lines = None):

        anchoring = np.array(in_slice['anchoring'])
        w=in_slice["width"]; h=in_slice["height"]

        o = anchoring[:3]
        u = anchoring[3:6]
        v = anchoring[6:]

        Bp = np.array([u,v,o])
        Bpp = np.array([[1/w, 0, 0], [0, 1/h, 0], [0, 0, 1]])
        #Bpp = np.array([[1/h, 0, 0], [0, 1/w, 0], [0, 0, 1]])

        # Using a third column for ones ..
        self.newimg_F = self.newimg_F.T
        tmp = np.zeros(self.newimg_F.shape,np.uint8)
        newCoordsF_2 = np.argwhere(tmp==0)
        if len(newCoordsF_2[0]) > 2:
            newCoordsF_2[:,2] = 1
        else:
            newCoordsF_2 = np.c_[newCoordsF_2, np.ones(len(newCoordsF_2))]
        newCoordsF_2 = np.array(newCoordsF_2,dtype=int)
        self.newCoordsF_2 = newCoordsF_2

        scaled_coords = Bpp.dot(self.newCoordsF_2.T)
        #Voxel_Coo = (Bp.dot(scaled_coords)).T
        Voxel_Coo = scaled_coords.T.dot(Bp)

        Voxel_Coo_trs = (self.Q_neuron2allen[0:3,0:3] @ Voxel_Coo.T).T + self.Q_neuron2allen[0:3,3]
        self.Voxel_Coo_trs = np.array(np.round(Voxel_Coo_trs),dtype = int)

        if lines is not None:
            self.lines = []
            for line in lines:
                scaled_coords = Bpp.dot(line.T)
                voxel_line = scaled_coords.T.dot(Bp)
                voxel_line_trs = (self.Q_neuron2allen[0:3,0:3] @ voxel_line.T).T + self.Q_neuron2allen[0:3,3]
                self.lines.append(list(np.array(np.round(voxel_line),dtype = int)))

    def ExtraPlots(self, fname, style = 'allen'): #Nest it in a save function

        newimg_PIL = Image.fromarray(self.newimg_F)
        newimg_PIL.save('{}/export/{}_transformed.png'.format(self.in_dir,self.fname.split('.')[0])) #./

        label_fig = fname.split('_SimpleSegmentation')[0] + '-Rainbow_2017'
        flat_infile = '{}/export/{}'.format(self.in_dir,label_fig) #./
        labels = json_to_flat(self.in_slice, self.annotation, parcelation = 'allen')

        if style == 'template':
            labels_extra = json_to_flat(self.in_slice, self.annotation, template = self.template, parcelation = 'template')
        else:
            labels_extra = None
        ao.ProduceSvg(self.newimg_F, self.in_dir, labels, labels_extra = labels_extra,
                   savefile = flat_infile + '_{}'.format(style), in_color = self.in_color)

        if self.annotation2 is not None:
            labels2 = json_to_flat(self.in_slice, self.annotation2, parcelation = 'YSK')
            ao.ProduceSvg(self.newimg_F, self.in_dir, labels2, labels_extra = labels_extra,
                       savefile = flat_infile + '_YSK', in_color = self.in_color)

    def register_sections(self, experiment_id, extra_plots = False, saveimg = False, revealed = False, style = 'allen'):

        in_dir = os.path.join(self.data_dir, self.experiment_id)
        find_mark_file = glob.glob(os.path.join(in_dir,'*_visualign.json'))
        path_to_images = f'{in_dir}/import'

        if len(find_mark_file) == 0:
            print('json file not found for this population. Skip.')
            return -1
        in_file = find_mark_file[0]
        try:
            # with open(os.path.join(in_dir,in_file)) as f:
            with open(in_file) as f:
                vafile = json.load(f)
        except:
            print('experiment {} has issues with the json files. Please modify it.'.format(self.experiment_id))
            return -1

        slice_list = np.argsort([vafile["slices"][i]['filename'] for i in range(len(vafile["slices"]))])
        for i in slice_list:
            in_slice = vafile["slices"][i]
            if "markers" not in in_slice.keys():
                print("markers field not present: moving to another slice!")
                continue

            if 'import\\' in in_slice["filename"]:
                fname = in_slice["filename"].split('import\\')[1].split('.')[0]
            else:
                fname = in_slice["filename"].split('.')[0]
            width = in_slice["width"]; height = in_slice["height"]
            self.fname = fname; self.in_slice = in_slice; self.width = width; self.height = height; self.in_dir = in_dir
            self.saveimg = saveimg
            dims = self.Vox_Mat.shape

            fname = fname + '_SimpleSegmentation_cleaned.png'
            if fname not in os.listdir(os.path.join(path_to_images)):
                fname = fname.split('_cleaned')[0] + '.png'
                if fname not in os.listdir(os.path.join(path_to_images)):
                    print('Error! This image is not available! Please provide an available image')
                    return -1

            try:
                self.in_img = np.array(Image.open('{}/{}'.format(path_to_images,fname)))
            except:
                print('Error! This image is not available! Please provide an available image')
                continue

            triangulation = VAL.triangulate(width,height,in_slice["markers"])
            self.newimg_F = np.zeros(self.in_img.shape,np.uint8)
            allCoords = np.argwhere(self.newimg_F==0)
            yx = VAL.forwardtransform_vec(triangulation,allCoords[:,1],allCoords[:,0])
            #print('forward transform completed')

            self.rev_transform_pixels(allCoords, yx)
            #print('non-linear reverse transform completed')

            if extra_plots == True:
                self.ExtraPlots(fname, style = style)

            # Convert pixels to voxels via the QuickNii transformation formula
            self.Pixel2Voxel(in_slice)
            #print('2D registration has been completed')

            # Map pixel intensity to voxel dictionaries separately for source and target and curate false source positives
            self.store_densities(dims)

            print('Slice {} from experiment {} has been registered.'.format(in_slice['filename'], self.experiment_id))

        return 1

    def store_densities(self, dims):

        a = self.Voxel_Coo_trs[:,0]>=dims[0]; b = self.Voxel_Coo_trs[:,1]>=dims[1]; c = self.Voxel_Coo_trs[:,2]>=dims[2]
        flag_pts = np.logical_or(a,np.logical_or(b,c))

        Voxel_Coo_trs_in_border = self.Voxel_Coo_trs[flag_pts==False,:]
        newCoordsF_in_border = self.newCoordsF_2[flag_pts==False,:]

        axon_mask = self.newimg_F[newCoordsF_in_border[:,0],newCoordsF_in_border[:,1]] == 255 # axon num
        axon_voxels = Voxel_Coo_trs_in_border[axon_mask,:]

        # for voxel_cord in np.unique(axon_voxels, axis = 0):
        #     v1,v2,v3 = voxel_cord
        #     if self.annotation[v1,v2,v3] == self.acr2id['[background]']: continue
        #     segment_num = len(np.where(axon_voxels == voxel_cord)[0])
        #     if tuple(voxel_cord) not in self.target_dict.keys():
        #         self.target_dict[tuple(voxel_cord)] = segment_num
        #     else:
        #         self.target_dict[tuple(voxel_cord)] += segment_num
        #!!! Warning!!  Alternative but may take longer ...
        for voxel_cord in axon_voxels:
            v1,v2,v3 = voxel_cord
            if self.annotation[v1,v2,v3] == self.acr2id['[background]']: continue
            if tuple(voxel_cord) not in self.target_dict.keys():
                self.target_dict[tuple(voxel_cord)] = 1
            else:
                self.target_dict[tuple(voxel_cord)] += 1

        unique_vals = np.unique(self.newimg_F)
        if len(unique_vals) > 2: # This is to ensure that we only estimate somas if a soma value is found

            soma_val = unique_vals[np.where(np.logical_and(unique_vals > 0, unique_vals < 255))[0]][0]
            soma_mask = self.newimg_F[newCoordsF_in_border[:,0],newCoordsF_in_border[:,1]] == soma_val # soma num
            soma_voxels = Voxel_Coo_trs_in_border[soma_mask,:]

            for voxel_cord in soma_voxels:  #!! Warning may have to change here too
                v1,v2,v3 = voxel_cord
                if self.annotation[v1,v2,v3] == self.acr2id['[background]']:
                    continue
                elif self.annotation[v1,v2,v3] != self.acr2id[self.soma_acr]:  # voxel outside of VPM has been labeled as soma, this should be re-translated as axon
                    if tuple(voxel_cord) not in self.target_dict.keys():
                        self.target_dict[tuple(voxel_cord)] = 1
                    else:
                        self.target_dict[tuple(voxel_cord)] += 1
                else:  # soma has been found in VPM
                    if tuple(voxel_cord) not in self.source_dict.keys():
                        self.source_dict[tuple(voxel_cord)] = 1
                    else:
                        self.source_dict[tuple(voxel_cord)] += 1

            # for voxel_cord in np.unique(soma_voxels, axis = 0):  #!! Warning may have to change here too
            #     v1,v2,v3 = voxel_cord
            #     if self.annotation[v1,v2,v3] == self.acr2id['[background]']:
            #         continue
            #     elif self.annotation[v1,v2,v3] != self.acr2id[self.soma_acr]:  # voxel outside of VPM has been labeled as soma, this should be re-translated as axon
            #         segment_num = len(np.where(axon_voxels == voxel_cord)[0])
            #         if tuple(voxel_cord) not in self.target_dict.keys():
            #             self.target_dict[tuple(voxel_cord)] = segment_num
            #         else:
            #             self.target_dict[tuple(voxel_cord)] += segment_num
            #     else:  # soma has been found in VPM
            #         segment_num = len(np.where(soma_voxels == voxel_cord)[0])
            #         if tuple(voxel_cord) not in self.source_dict.keys():
            #             self.source_dict[tuple(voxel_cord)] = segment_num
            #         else:
            #             self.source_dict[tuple(voxel_cord)] += segment_num

    def split_hemisphere(self, objective = 'target'):

        right_dict = OrderedDict()
        left_dict = OrderedDict()

        if objective == 'target':
            for target_voxels in self.target_dict.keys():
                v1,v2,v3 = target_voxels
                fl_v3 = deepcopy(v3)
                if fl_v3 > self.annotation.shape[2]//2:  # needs flipping ...
                    fl_v3 = 1140 - v3
                    left_dict[(v1,v2,fl_v3)] = self.target_dict[target_voxels]
                else:
                    right_dict[(v1,v2,v3)] = self.target_dict[target_voxels]
        elif objective == 'source':
            for source_voxels in self.source_dict.keys():
                v1,v2,v3 = source_voxels
                fl_v3 = deepcopy(v3)
                if fl_v3 > self.annotation.shape[2]//2:  # needs flipping ...
                    fl_v3 = 1140 - v3
                    left_dict[(v1,v2,fl_v3)] = self.source_dict[source_voxels]
                else:
                    right_dict[(v1,v2,v3)] = self.source_dict[source_voxels]

        return left_dict, right_dict

    def get_area_stats(self, pop_dict):

        un_num = lambda x : x.split('1')[0].split('2/3')[0].split('4')[0].split('5')[0].split('6a')[0].split('6b')[0]
        id2acr = {val:key for key,val in self.acr2id.items()}

        # Round 0: gathering regional stats for the populations in the SSp and SSs groups
        neurite_stats = OrderedDict()
        proj_class = OrderedDict()
        area_layer_dict = OrderedDict()

        for pop_name in pop_dict.keys():

            neurite_stats[pop_name] = {}

            for voxel_cords,voxel_density in pop_dict[pop_name].items():
                v1,v2,v3 = voxel_cords
                area = id2acr[self.annotation[v1,v2,v3]]
                if area not in neurite_stats[pop_name].keys():
                    neurite_stats[pop_name][area] = 0
                neurite_stats[pop_name][area] += voxel_density

        for pop_name in pop_dict.keys():
            ssp_cnt = 0; sss_cnt = 0
            for area,counts in neurite_stats[pop_name].items():
                if 'SSp' in area:
                    ssp_cnt += counts
                elif 'SSs' in area:
                     sss_cnt += counts
            if ssp_cnt > sss_cnt:
                proj_class[pop_name] = 'SSp'
            else:
                proj_class[pop_name] = 'SSs'

        # Round 2: counting segmented pixels per layer for the SSp and SSs groups
        area_layer_dict = OrderedDict([('SSp', OrderedDict()),('SSs', OrderedDict())])

        for pop_name in neurite_stats.keys():
            for area,counts in neurite_stats[pop_name].items():
                if proj_class[pop_name] in area:
                    if pop_name not in area_layer_dict[proj_class[pop_name]].keys():
                        area_layer_dict[proj_class[pop_name]][pop_name] = {'L1': 0 , 'L2/3': 0, 'L4': 0, 'L5': 0, 'L6a': 0,'L6b': 0}
                    layer = get_layer_from_area(area)
                    area_layer_dict[proj_class[pop_name]][pop_name][layer] += counts

        area_layer_dict['SSp'] = pd.DataFrame(area_layer_dict['SSp']).T
        area_layer_dict['SSs'] = pd.DataFrame(area_layer_dict['SSs']).T

        area_layer_dict['SSp'] = ((area_layer_dict['SSp'].T)/area_layer_dict['SSp'].sum(axis = 1)).T
        area_layer_dict['SSp'] = area_layer_dict['SSp'].reindex(sorted(area_layer_dict['SSp'].columns)[::-1], axis=1)
        area_layer_dict['SSs'] = ((area_layer_dict['SSs'].T)/area_layer_dict['SSs'].sum(axis = 1)).T
        area_layer_dict['SSs'] = area_layer_dict['SSs'].reindex(sorted(area_layer_dict['SSs'].columns)[::-1], axis=1)

        return area_layer_dict, neurite_stats

    def image_to_x3d(self, in_point_cloud, in_colors = [1.0, 0.0, 0.0], labelIndex = 2, filename = "", savefile = None):

        in_point_cloud_arr = np.array(list(in_point_cloud.keys())) # convert to array
        img = np.zeros(self.annotation.shape, dtype=np.uint8)
        img[in_point_cloud_arr[:,0],in_point_cloud_arr[:,1],in_point_cloud_arr[:,2]] = labelIndex
        img[img == 0] = 1

        subimg = []
        k = 0
        size = img.shape
        for di in [0,1]:
          for dj in [0,1]:
            for dk in [0,1]:
                subimg.append(img[di:size[0]-1+di:2,dj:size[1]-1+dj:2,dk:size[2]-1+dk:2])

        Q = CAS.convertAllenSpace(['um(10)','PIR','corner'],['um','PIR','corner'])

        in_colors = np.array(in_colors)/255
        in_colors_mod = in_colors + 0.1
        in_colors_mod[in_colors_mod > 1] = 1
        colors = {
            1:'0.4 0.4 0.4',
            2: '{} {} {}'.format(in_colors[0],in_colors[1],in_colors[2]),
            3: '{} {} {}'.format(in_colors_mod[0],in_colors_mod[1],in_colors_mod[2])
        }
        print(colors)

        sizes = {
            1:'15 15 15',
            2:'20 20 20',
            3:'60 60 60'
        }

        jitter = [50,0,0];
        x3dScene = ["<X3D><Scene>"]
        for c,clr in colors.items():
            x3dScene.append('\n'.join([
                "<Shape><Appearance DEF='\"color{}\"'><Material diffuseColor='{}' specularColor='{}'></Material>".format(c,clr,clr),
                "</Appearance>",
                "</Shape>"
            ]))

        allI,allJ,allK = np.nonzero(np.sum([subim == labelIndex for subim in subimg], axis = 0))
        for i in range(len(allI)):
            coord = [str(round(20*(v+1)+jitter[d]*np.random.uniform(-1,1))) for d,v in enumerate([allI[i],allJ[i],allK[i]])]
            x3dScene.append('\n'.join([
                "<Transform translation='{}'><Shape><Appearance USE='\"color{}\"'></Appearance>".format(' '.join(coord),labelIndex),
                "<Box size='{}'></Box>".format(sizes[labelIndex]),
                "</Shape></Transform>"
            ]))
        x3dScene.append("</Scene></X3D>")

        contents = '\n'.join(x3dScene)
        sbaCommand = {
          "method": "Composer.import",
          "params": {
            "@type": 'vectorgraphics',
            "name": "{}".format(filename),
            "mediaType": "model/x3d+xml",
            "space": 'sba:ABA_v3(um,PIR,corner)',
            "contents": contents
          }
        }

        if savefile is not None:
            with open('Saves/{}_bas(sba.ABA_v3(um,PIR,corner)).x3d'.format(filename),'wt') as fp:
                fp.write('\n'.join(x3dScene))

        return x3dScene, sbaCommand




#******
