
from cfg import *
from flatmapper import FlatMapper
sys.path.append('atlasoverlay')
import atlasoverlay as ao

hex2rgb = lambda x : tuple(int(x.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))

class BrainPlotter:
    def __init__(self, a = 1, atlas_overlay_dir = './', flatmapper_dir = './', data_dir = './',
                 target_color = [[0,0,255],[0,255,0]], hemisphere = None):
        self.a = a


def json_to_flat(cur_slice, atlas, template = None, parcelation = 'allen', outdir = '/tmp', indir = 'Saves/'):

    palette={} # dictionary to store the data to be plotted?
    if parcelation == 'allen':
        label_name = 'labels_itk_allen.txt' #'labels.txt' #
    else:
        label_name = 'labels_itk_ysk.txt'

    with open("{}/{}".format(indir, label_name)) as f: # this file contains the correspondence between the regions and their indexs
        for line in f: # loop through indxs of areas
            lbl=re.match(r'\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)',line) # no idea what this does
            # I think this is a boolean that checks if each line conforms to a given format
            # if it does, then it's True and the values are incorporated to the palette variable in the line below
            if lbl:
                palette[int(lbl[1])]=((int(lbl[2]),int(lbl[3]),int(lbl[4])))

    o,u,v = np.reshape(cur_slice['anchoring'],(3,3))* 2.5
    slice_name = cur_slice['filename']
    w = int(np.linalg.norm(u)) + 1 # get the 2-norm of vector u, which would be the x coord
    h = int(np.linalg.norm(v)) + 1 # get the 2-norm of vector v (euclidean distance), the y coord

    # image = PIL.Image.new("RGB",(w,h))
    image = Image.new("RGB",(w,h))

    if parcelation == 'template':
        for y in range(h): # loop through the norm of u
            for x in range(w): # loop through the norm of v
                i,j,k = o + u*x/w + v*y/h # get the coordinates for the plane
                if 0 <= i < 1140 and 0 <= j < 1320 and 0 <= k < 800: # if within the confines of the array
                    gray_temp = np.round(template[1319-int(j)][799-int(k)][1139-int(i)]/2).astype(int)
                    image.putpixel((x,y),((gray_temp),(gray_temp),(gray_temp))) # get the rgb for an id
    else:
        for y in range(h): # loop through the norm of u
            for x in range(w): # loop through the norm of v
                i,j,k = o + u*x/w + v*y/h # get the coordinates for the plane
                if 0 <= i < 1140 and 0 <= j < 1320 and 0 <= k < 800: # if within the confines of the array
                    image.putpixel((x,y),palette[atlas[1319-int(j)][799-int(k)][1139-int(i)]]) # get the rgb for an id

    save_name = slice_name.split('.')[0]
    image.save(f"{outdir}/{save_name}.png","PNG")
#     IPython.display.display(image) # plot it!
    image_array = np.asarray(image)

    return image_array

def corticalProjection(projectionType,dataVolume,aggregateFunction,savePng=None,saveNifti=None):
  mapper = cortical_map.CorticalMap(projectionType,dataVolume.shape)
  proj = mapper.transform(dataVolume, agg_func = aggregateFunction)
  if savePng:
    im = Image.fromarray(proj.astype(np.uint8))
    im.save(savePng)
  if saveNifti:
    # for inspection with ITK-snap
    nii = nibabel.Nifti1Image(proj,np.eye(4))
    nibabel.save(nii,saveNifti)
  return proj

def selectiveCorticalProjection(projectionType,dataVolume,aggregateFunction, labelVolume,labelSelection,savePng=None,saveNifti=None):
  mapper = cortical_map.CorticalMap(projectionType,dataVolume.shape)
  proj = mapper.selectiveTransform(dataVolume,labelVolume,labelSelection, agg_func = aggregateFunction)
  if savePng:
    im = Image.fromarray(proj.astype(np.uint8))
    im.save(savePng)
  if saveNifti:
    # for inspection with ITK-snap
    nii = nibabel.Nifti1Image(proj,np.eye(4))
    nibabel.save(nii,saveNifti)
  return proj

# AGGREGATE FUNCTIONS
def layerMapFunc(regionsByLayer):
  def AF(arr):
    nonzero = arr[arr>0]
    if len(nonzero):
      hasLayer = [0,0,0,0,0]
      for id in nonzero:
        if id in regionsByLayer['Layer1']:
          hasLayer[0] = 1
        elif id in regionsByLayer['Layer2_3']:
          hasLayer[1] = 1
        elif id in regionsByLayer['Layer4']:
          hasLayer[2] = 1
        elif id in regionsByLayer['Layer5']:
          hasLayer[3] = 1
        elif id in regionsByLayer['Layer6']:
          hasLayer[4] = 1
      return hasLayer[0]+2*hasLayer[1]+4*hasLayer[2]+8*hasLayer[3]+16*hasLayer[4]
    return 0
  return AF

def nonlayerMapFunc(regionsByLayer):
  def AF(arr):
    nonzero = arr[arr>0]
    if len(nonzero):
      hasLayer = [0,0,0,0,0]
      for id in nonzero:
        if not (id in regionsByLayer['Layer1'] or id in regionsByLayer['Layer2_3'] or id in regionsByLayer['Layer4'] or id in regionsByLayer['Layer5'] or id in regionsByLayer['Layer6']):
          return id
    return 0
  return AF

def firstNonzeroFunc():
  def AF(arr):
    nonzero = arr[arr>0]
    if len(nonzero):
      return nonzero[0]
    return 0
  return AF

def firstElemFunc():
  def AF(arr):
    if len(arr):
      return arr[0]
    return 0
  return AF

def lastElemFunc():
  def AF(arr):
    if len(arr):
      return arr[-1]
    return 0
  return AF

def selectAreaFunc(ancestorsById,allowedParentIds):
  # Find out which areas have children
  hasChildren = set()
  for id,ancestors in ancestorsById.items():
    if len(ancestors) > 1:
      parent = ancestors[1]
      hasChildren.add(parent)

  def AF(arr):
    nonzero = arr[arr>0]
    if len(nonzero):
      x = []
      for id in nonzero:
        ancestors = ancestorsById[str(id)]
        ok = False
        for pid in allowedParentIds:
          if pid in ancestors:
            ok = True
        if ok:
          if ancestors[0] in hasChildren:
            x.append(ancestors[0])
          else:
            x.append(ancestors[1])
      if len(x):
        unique, counts = np.unique(x, return_counts=True)
        return unique[counts.argmax()]

    return 0

  return AF


def projectAnnotation(projectionType, annotationVolume, ancestorsById, allowedParentIds, savePng=None,saveNifti=None):
  idImage  = corticalProjection(projectionType,annotationVolume,selectAreaFunc(ancestorsById,allowedParentIds),savePng=savePng,saveNifti=saveNifti)
  return idImage


def getAnnotationOverlay(projectionType,lineColor='#000', lineWidth='3',labelColor='#000',labelGlow='#AAA',
                         data_dir = './', annot = 'allen', hemisphere = None):
    if hemisphere is None:
        hemi_str = ''
    else:
        hemi_str = ',lh'
    if annot == 'allen':
         annotationOverlay = os.path.join(data_dir,'annotation_10({}{}).svg'.format(projectionType,hemi_str))
    else:
         annotationOverlay = os.path.join(data_dir,'YSK_annotation_smooth_10({}).svg'.format(projectionType))

    parser = etree.XMLParser(remove_blank_text=True)
    tree = etree.parse(annotationOverlay,parser)
    groups = tree.xpath('//*[name()="g"]')
    for g in groups:
        id = g.get('id')
        if id and id[0]=='#':
            g.set('stroke',lineColor)
            g.set('stroke-width',lineWidth)

    return etree.tostring(tree).decode('utf-8')


def set_color(input_mat, in_color, cmap_bg):
    # one idea is to overlay two colourmaps
    max_input = np.max(input_mat)
    unique_vals = len(np.unique(input_mat))
    mask = input_mat > 0
    input_mat_rgb = cmap_bg(input_mat, bytes = True)

    # Create a colormap based on the input color ..
    if unique_vals == 2:
        if max_input > 0:   # There are labeled points in the cortex
            input_mat_rgb[:,:,0][mask] = in_color[0]
            input_mat_rgb[:,:,1][mask] = in_color[1]
            input_mat_rgb[:,:,2][mask] = in_color[2]
    else:
        in_color_nrm = np.array(in_color)/255
        # color_range =  np.array([in_color_nrm/4, in_color_nrm/3, \
        #                 in_color_nrm/2, in_color_nrm])
        # !! Try this !!!
        color_range =  np.array([in_color_nrm/4, in_color_nrm/3 + [0.1,0.1,0.1], \
                        in_color_nrm/2 + [0.2,0.2,0.2], in_color_nrm + [0.3,0.3,0.3]])


        color_range[color_range > 1] = 1
        cmap_fg = LinearSegmentedColormap.from_list('random', color_range, N = unique_vals)
        input_mat_fg = cmap_fg(input_mat, bytes = True)

        if max_input > 0:   # There are labeled points in the cortex
            input_mat_rgb[:,:,0][mask] = input_mat_fg[:,:,0][mask]
            input_mat_rgb[:,:,1][mask] = input_mat_fg[:,:,1][mask]
            input_mat_rgb[:,:,2][mask] = input_mat_fg[:,:,2][mask]

    return input_mat_rgb


def format_func(value, tick_number):
    return str(np.around(value * 0.001, 3))


def plot_flatmap(voxel_to_value_dict, proj_type = 'dorsal_flatmap', template = None, annot_file = None,
                 savefile = None, data_dir = './', annot = 'allen', atlas_overlay_dir = './', flatmapper_dir = './',
                 target_color = [[0,0,255],[0,255,0]], hemisphere = None):

    start = time.time()

    mapper = FlatMapper(flatmapper_dir)
    if template is None:
        if hemisphere is None or annot == 'YSK':  # YSK is temporary because we currently do not support half YSK flatmaps
            hemi_str = ''
        else:
            hemi_str = '_lh'
        template10 = np.array(Image.open(os.path.join(data_dir, 'template_{}{}_gc.png'.format(proj_type,hemi_str))))
    else:
        template10 = template
    if annot_file is None:
        try:
            annotationOverlay = getAnnotationOverlay(proj_type, lineWidth='2', labelColor='#000', lineColor='#000',
                                                     labelGlow='#444', data_dir = data_dir, annot = annot,
                                                     hemisphere = hemisphere)
        except:
            print('Error! Annotation overlay file is either not present or cannot be parsed.')
            # In a future version I should just generate the overlay from scratch here ...
            return -1
    else:
        annot_fname = annot_file.split('/')[len(annot_file.split('/'))-1]
        if annot_fname in os.listdir(data_dir):
            annotationOverlay = annot_file
        else:
            print('Error! Annotation overlay file is either not present or cannot be parsed.')
            # In a future version I should just generate the overlay from scratch here ...
            return -1

    #%% initialize colormaps for rgb conversion
    if len(np.shape(target_color)) == 1:
        target_color = [target_color]
    cmap_mask     = plt.get_cmap('gray')

    # this consumes a few GBs of memory
    if proj_type == 'dorsal_flatmap':
        projectionType = 'flatmap_dorsal'
    else:
        projectionType = proj_type
    mapper.init(projectionType)

    unravel_voxels = np.array([(pop_idx,pop_cord,pop_val) for pop_idx, pop_vals in enumerate(voxel_to_value_dict.values())
                               for pop_cord,pop_val in pop_vals.items()])
    point_cloud_labels = np.asarray(list(unravel_voxels[:,0]))
    point_cloud_voxel_cords = np.asarray(list(unravel_voxels[:,1]))
    point_cloud_vals   = np.asarray(list(unravel_voxels[:,2]))

    point_cloud_flat_cords = np.array(mapper.projectPoints(point_cloud_voxel_cords))

    for pop_idx in np.unique(point_cloud_labels):

        pop_indices = np.where(point_cloud_labels == pop_idx)
        pixel_vals = point_cloud_vals[pop_indices]
        pixel_cords = point_cloud_flat_cords[pop_indices]
        trs_pd = np.zeros((template10.shape[0],template10.shape[1]),dtype=np.uint8)
        trs_pd[pixel_cords[:,0],pixel_cords[:,1]] = pixel_vals[:]
        trs_pd_nrm    = trs_pd/(np.max(trs_pd)*1.0)
        if pop_idx > len(target_color) - 1:
            target_color.append(np.random.randint(0,high = 255,size = 3))
        rgb_pd = set_color(trs_pd_nrm, target_color[pop_idx], cmap_mask)
        if pop_idx == 0:
            rgb_export  = deepcopy(rgb_pd)
            trs_old     = deepcopy(trs_pd)
        else:
            rgb_export[trs_pd > trs_old] = rgb_pd[trs_pd > trs_old]
            trs_old     = np.maximum(trs_pd, trs_old)

    fig = plt.figure(figsize=(template10.shape[1]/240,template10.shape[0]/240))
    ax = plt.axes([0,0,1,1])
    bgImage = template10
    pos = ax.imshow(bgImage,plt.get_cmap('gray'))
    plt.axis('off')
    svgFig = ao.matplot2svg(fig)
    print(rgb_export.shape)
    print(template10.shape)
    svg = ao.array2svg(rgb_export)
    ao.atlasOverlay(svgFig, ax, svg, in_path = atlas_overlay_dir)
    ao.atlasOverlay(svgFig, ax, annotationOverlay, in_path = atlas_overlay_dir)
    ao.displaySvgFigure(svgFig)
    svg = ao.stringifySvgFigure(svgFig)
    if savefile is not None:
        with open(savefile.split('.')[0]+'.svg','wt') as fp:
            fp.write(svg)

    mapper.done()
    end = time.time()
    print('time elapsed : {}'.format((end-start)/60))

    return rgb_export


def Subcortical_Map(point_cloud, pixel_vals, annotation, template, acr2id, area_name = 'VPM', sel_axis = 'IR',
                    orient = 'left', style = 'max', section = None):

    plane_to_id = {'PI': 2,'IR': 0,'RP': 1}

    area_mask = annotation==acr2id[area_name] # give me VPM for instance
    area_loci = np.where(area_mask)
    x_min,x_max = np.min(area_loci[0]), np.max(area_loci[0])
    y_min,y_max = np.min(area_loci[1]), np.max(area_loci[1])
    z_min,z_max = np.min(area_loci[2]), np.max(area_loci[2])
    if orient == 'left':
        z_midpoint = z_min + int(np.round(1*(z_max-z_min)/3)) -1   # taking the midth along the left-right axis
        z_max = z_midpoint
    elif orient == 'right':
        z_midpoint = z_min + int(np.round(3*(z_max-z_min)/4)) -1   # taking the midth along the left-right axis
        z_min = z_midpoint
        z_max = z_max - 1
    return_coos = (x_min, x_max, y_min,y_max,z_min,z_max)
    area_volume = deepcopy(annotation[x_min:x_max+1,y_min:y_max+1,z_min:z_max+1])
    area_intensity = deepcopy(template[x_min:x_max+1,y_min:y_max+1,z_min:z_max+1])

    soma_volume = np.zeros(np.shape(annotation), dtype = 'uint8')
    soma_volume[point_cloud[:,0],point_cloud[:,1],point_cloud[:,2]] = pixel_vals
    soma_volume_specific = deepcopy(soma_volume[x_min:x_max+1,y_min:y_max+1,z_min:z_max+1])

    soma_volume_cords = np.nonzero(soma_volume_specific)
    if len([val for val in soma_volume_specific[soma_volume_cords] if val > 0 and val < 1]) > 0:
          soma_volume_specific *= 100

    x_midpoint = int(np.round((x_max-x_min)/2))-1   # taking the midth along the left-right axis
    y_midpoint = int(np.round((y_max-y_min)/2))-1   # taking the midth along the left-right axis
    z_midpoint = int(np.round((z_max-z_min)/2))-1   # taking the midth along the left-right axis

    #************
    max_val = np.max(area_volume)*2
    area_volume[area_volume == acr2id[area_name]] = max_val

    area_volume_clone = deepcopy(area_volume)

    if style == 'solo' or style == 'median':
        area_volume[area_volume != max_val] = 0
    if style == 'median':
        if plane_to_id[sel_axis] == 0:
            area_volume_mid = area_volume_clone[x_midpoint,:,:]
        elif plane_to_id[sel_axis] == 1:
            area_volume_mid = area_volume_clone[:,y_midpoint,:]
        elif plane_to_id[sel_axis] == 2:
            area_volume_mid = area_volume_clone[:,:,z_midpoint]
    elif style == 'max' and section is not None:
        if plane_to_id[sel_axis] == 0:
            area_volume_tmp = area_volume_clone[section - x_min,:,:]
            soma_volume = soma_volume_specific[section - x_min,:,:]
            area_intensity = area_intensity[section - x_min,:,:]
        elif plane_to_id[sel_axis] == 1:
            area_volume_tmp = area_volume_clone[:,section - y_min,:]
            soma_volume = soma_volume_specific[:,section - y_min,:]
            area_intensity = area_intensity[:,section - y_min,:]
        elif plane_to_id[sel_axis] == 2:
            area_volume_tmp = area_volume_clone[:,:,section - z_min]
            soma_volume = soma_volume_specific[:,:,section - z_min]
            area_intensity = area_intensity[:,:,section - z_min]
    #***********
    if section is None:
        area_volume_tmp = np.max(area_volume, axis = plane_to_id[sel_axis])
        area_intensity = np.max(area_intensity, axis = plane_to_id[sel_axis])
    if style == 'median':
        area_volume_mid[area_volume_tmp==max_val] = max_val
        area_volume_tmp = area_volume_mid

    unique_annots = np.unique(area_volume_tmp)
    for idx,annot in enumerate(unique_annots):
        area_volume_tmp[area_volume_tmp==annot] = (idx + 1)*4

    area_volume = np.array(area_volume_tmp, dtype = 'uint8')
    area_intensity = np.array(area_intensity, dtype = 'uint8')

    if len(soma_volume_cords[0]) == 0:
        print('Volume is empty. Please provide new data.')
        soma_volume = np.zeros(area_intensity.shape, dtype = 'uint8')
    else:
        if section is None:
            soma_volume = np.array(np.max(soma_volume_specific, axis = plane_to_id[sel_axis]), dtype = 'uint32')*355

    # maybe return_coos too
    return soma_volume, area_volume, area_intensity, return_coos

def plot_plane(voxel_to_value_dict, annotation, template, acr2id, source_area, in_color = [[0,0,255],[0,255,0]], in_path = './',
                sel_axis = 'IR', orient = 'left', savefile = None, style = 'max', section = None):
    # Three potential flatmapping strategies:
    # 'max': taking the maximum intensity of everything including anatomical borders
    # 'median': taking maximum intensity only for the source border, and overlaying it with the middle anatomical slice for the rest
    # 'solo': taking only the source border into account, and not including any other border
    cmap_pd  = plt.get_cmap('hot')
    cmap_gr  = plt.get_cmap('gray')

    if len(np.shape(in_color)) == 1:
        in_color = [in_color]

    for pop_idx,pop_voxels in enumerate(voxel_to_value_dict.values()):

        pop_vals = np.asarray(list(pop_voxels.values()))
        pop_cords = np.asarray(list(pop_voxels.keys()))
        if style == 'slice':
            input_plane, plane_annotation, plane_intensity = Slice_Maker(pop_cords, pop_vals,
                                                                          annotation, template, acr2id,
                                                                          sel_axis, section = section)
        else:
            input_plane, plane_annotation, plane_intensity, return_coos = Subcortical_Map(pop_cords, pop_vals,
                                                                                          annotation, template, acr2id,
                                                                                          source_area, sel_axis, orient,
                                                                                          style, section = section)
        input_plane_nrm = input_plane/(np.max(input_plane)*1.0)
        if pop_idx > len(in_color)-1:
            in_color.append(np.random.randint(0,high = 255,size = 3))
        try:
            if input_plane == -1:
                continue
        except:
            a = 1

        plane_mix = set_color(input_plane_nrm, in_color[pop_idx], cmap_gr)
        if pop_idx == 0:
            rgb_export = plane_mix
            input_plane_old = deepcopy(input_plane)
        else:
            rgb_export[input_plane > input_plane_old] = plane_mix[input_plane > input_plane_old]
            input_plane_old = np.maximum(input_plane, input_plane_old)

    fig = plt.figure(figsize=(15, 6))
    ax1   = plt.axes()
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(format_func))
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(format_func))
    pos = ax1.imshow(plane_intensity, cmap_gr)
    plt.axis('off')
    svgFig = ao.matplot2svg(fig)
    svg = ao.array2svg(rgb_export)
    ao.atlasOverlay(svgFig, ax1, svg, in_path = in_path)
    ao.atlasOverlay(svgFig, ax1, plane_annotation, in_path = in_path, strokeWidth=1, contourSmoothness=1)
    ao.displaySvgFigure(svgFig)

    if savefile is not None:
        ao.saveSvgFigure(svgFig, savefile.split('.')[0] + '.svg')

    return input_plane, plane_annotation, plane_intensity #, return_coos


def Slice_Maker(label_volume_specific, annotation, template, acr2id, sel_axis = 'IR', section = None):

    plane_to_id = {'PI': 2,'IR': 0,'RP': 1}

    return_coos = (0,annotation.shape[0]-1,0,annotation.shape[1]-1,0,annotation.shape[2]-1)
    x_min,x_max = return_coos[0],return_coos[1]
    y_min,y_max = return_coos[2],return_coos[3]
    z_min,z_max = return_coos[4],return_coos[5]

    area_volume_clone = deepcopy(annotation)
    area_intensity_clone = deepcopy(template)
    label_volume_clone = deepcopy(label_volume_specific)
    label_volume_cords = np.nonzero(label_volume_specific)

    if len(label_volume_cords[0]) == 0:
        print('Volume is empty. Please provide new data.')
#         return -1,-1,-1,-1

    x_midpoint = int(np.round((x_max-x_min)/2))-1   # taking the midth along the left-right axis
    y_midpoint = int(np.round((y_max-y_min)/2))-1   # taking the midth along the left-right axis
    z_midpoint = int(np.round((z_max-z_min)/2))-1   # taking the midth along the left-right axis

    if plane_to_id[sel_axis] == 0:
        if section is None:
            area_volume_mid = area_volume_clone[x_midpoint,:,:]
            area_intensity_mid = area_intensity_clone[x_midpoint,:,:]
            soma_volume_mid = label_volume_clone[x_midpoint,:,:]
        else:
            area_volume_mid = area_volume_clone[section,:,:]
            area_intensity_mid = area_intensity_clone[section,:,:]
            soma_volume_mid = label_volume_clone[section,:,:]
    elif plane_to_id[sel_axis] == 1:
        if section is None:
            area_volume_mid = area_volume_clone[:,y_midpoint,:]
            area_intensity_mid = area_intensity_clone[:,y_midpoint,:]
            soma_volume_mid = label_volume_clone[:,y_midpoint,:]
        else:
            area_volume_mid = area_volume_clone[:,section,:]
            area_intensity_mid = area_intensity_clone[:,section,:]
            soma_volume_mid = label_volume_clone[:,section,:]
    elif plane_to_id[sel_axis] == 2:
        if section is None:
            area_volume_mid = area_volume_clone[:,:,z_midpoint]
            area_intensity_mid = area_intensity_clone[:,:,z_midpoint]
            soma_volume_mid = label_volume_clone[:,:,z_midpoint]
        else:
            area_volume_mid = area_volume_clone[:,:,section]
            area_intensity_mid = area_intensity_clone[:,:,section]
            soma_volume_mid = label_volume_clone[:,:,section]

    unique_annots = np.unique(area_volume_mid)
    for idx,annot in enumerate(unique_annots):
        area_volume_mid[area_volume_mid==annot] = (idx + 1)#*5

    cmap = plt.get_cmap('gray')
    area_volume = cmap(area_volume_mid, bytes = True)
    area_volume = np.array(area_volume_mid, dtype = 'uint8')
    area_intensity = np.array(area_intensity_mid, dtype = 'uint32')
    if section is None:
        label_volume = np.array(np.max(label_volume_clone, axis = plane_to_id[sel_axis]), dtype = 'uint32')*355
    else:
        label_volume = np.array(soma_volume_mid, dtype = 'uint32')*355

    return label_volume, area_volume, area_intensity
