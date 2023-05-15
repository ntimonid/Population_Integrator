import re
import json
import numpy; import numpy as np
import nrrd
import os
import copy
import gzip


# unitCode:
# - string that defines unit, with in-between brackets a multiplier in one or more dimensions.
#   The string can be m / mm / um.
#   Examples: mm, mm(25), um(10,10,200)
# orientationCode:
# - set of three letters indicating the anatomical orientation of the positive x, y and z coordinate axes.
#   One axis must be assigned (R)ight or (L)eft,
#   another axis must be assigned (A)nterior or (P)osterior,
#   and the remaining axis must be assigned (S)uperior or (I)nferior.
#   Examples: PIR, LIP, RAS
# originCode:
# - string that indicates the landmark used as the origin.
#   For the Allen atlas volume, this can be 'center', 'corner', or 'ac (anterior commissure)'

def parseUnitCode(unitCode):
  m = re.match('(m|mm|um)(\(.*\))?$',unitCode)
  unit = m.group(1)
  group2 = m.group(2)
  multiplier = None
  if group2:
    multiplier = []
    m = re.match('\((.*)\)$',group2)
    group1 = m.group(1)
    groups = []
    if group1:
      groups = group1.split(',')
      for i,g in enumerate(groups):
        multiplier.append(float(g))
  return unit,multiplier

def parseOrientationCode(orientationCode):
  dims = None
  m = re.match('(R|L)(A|P)(S|I)$',orientationCode)
  if m:
    dims = [0,1,2]
  else:
    m = re.match('(R|L)(S|I)(A|P)$',orientationCode)
    if m:
      dims = [0,2,1]
    else:
      m = re.match('(A|P)(R|L)(S|I)$',orientationCode)
      if m:
        dims = [1,0,2]
      else:
        m = re.match('(A|P)(S|I)(R|L)$',orientationCode)
        if m:
          dims = [1,2,0]
        else:
          m = re.match('(S|I)(R|L)(A|P)$',orientationCode)
          if m:
            dims = [2,0,1]
          else:
            m = re.match('(S|I)(A|P)(R|L)$',orientationCode)
            if m:
              dims = [2,1,0]

  flip = None
  if m:
    flip = [0,0,0]
    targetOrientation = 'RAS'
    for i in range(0,3):
      flip[i] = 1 if m.group(1+i)==targetOrientation[dims[i]] else -1
  return dims,flip

def parseOriginCode(originCode,orientationCode):
  Annotation25_RAS_shape = numpy.array([456, 528, 320],float)
  targetOrigin_mm_RAS_center = Annotation25_RAS_shape/2*25e-3
  origin_mm_RAS = None
  m = re.match('center|corner|ac',originCode)
  if m:
    group = m.group(0)
    if group == 'center':
      origin_mm_RAS = targetOrigin_mm_RAS_center
    else:
      if group == 'ac':
        origin_voxel_RAS = numpy.array([228, 313, 113],float)
        origin_mm_RAS = origin_voxel_RAS*25e-3
      else:
        if group == 'corner':
          dims,flip = parseOrientationCode(orientationCode)
          origin_mm_RAS = [0,0,0]
          for i in range(0,3):
            if flip[i]<0:
              origin_mm_RAS[dims[i]] = Annotation25_RAS_shape[dims[i]]*25e-3
  return origin_mm_RAS


def getAffine_unit(unit,multiplier):
  toMm = 1
  if unit == 'm': toMm = 1e3
  else:
    if unit == 'um': toMm = 1e-3

  if multiplier is None:
    multiplier = [1,1,1]
  else:
    numel = len(multiplier)
    if numel<3:
      lastValue = multiplier[numel-1]
      while numel<3:
        multiplier.append(lastValue)
        numel += 1
  return numpy.array([
    [toMm*multiplier[0], 0, 0, 0],
    [0, toMm*multiplier[1], 0, 0],
    [0, 0, toMm*multiplier[2], 0],
    [0, 0, 0, 1]
  ],float)

def getAffine_orientation(dims,flip):
  A = numpy.array([
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 1]
  ],float)
  # A must be such that X_RAS = A*X
  A[dims[0],0] = flip[0]
  A[dims[1],1] = flip[1]
  A[dims[2],2] = flip[2]
  return A

def getAffine_origin(origin_mm_RAS):
  origin_mm_RAS_center = parseOriginCode('center','RAS')
  #print('origin_mm_RAS_center',origin_mm_RAS_center)
  origin_shift = numpy.array(origin_mm_RAS,float)-numpy.array(origin_mm_RAS_center,float)
  #print('origin_mm_RAS',origin_mm_RAS)
  return numpy.array([
    [1, 0, 0, origin_shift[0]],
    [0, 1, 0, origin_shift[1]],
    [0, 0, 1, origin_shift[2]],
    [0, 0, 0, 1]
  ],float)


def toAllen_mm_RAS_center(unitCode,orientationCode,originCode):
  unit,multiplier = parseUnitCode(unitCode)
  A_unit = getAffine_unit(unit,multiplier)
  dims,flip = parseOrientationCode(orientationCode)
  A_reorient = getAffine_orientation(dims,flip)
  origin = parseOriginCode(originCode,orientationCode)
  A_origin = getAffine_origin(origin)
  return A_origin @ A_reorient @ A_unit

def test_toAllen_mm_RAS_center():
  unit,multiplier = parseUnitCode('m')
  print('unit, multiplier',unit,multiplier)
  assert(unit == 'm')
  assert(multiplier is None)
  unit,multiplier = parseUnitCode('mm(25)')
  print('unit, multiplier',unit,multiplier)
  assert(unit == 'mm')
  assert(len(multiplier) == 1 and multiplier[0] == 25)
  unit,multiplier = parseUnitCode('um(10,10,200)')
  print('unit, multiplier',unit,multiplier)
  assert(unit == 'um')
  assert(len(multiplier) == 3 and multiplier[0] == 10 and multiplier[1] == 10 and multiplier[2] == 200)
  A_unit = getAffine_unit(unit,multiplier)
  A_assert = numpy.array([
    [1e-3*10, 0, 0, 0],
    [0, 1e-3*10, 0, 0],
    [0, 0, 1e-3*200, 0],
    [0, 0, 0, 1]
  ],float)
  assert(numpy.all(A_unit == A_assert))

  dims,flip = parseOrientationCode('PIR')
  print(dims,flip)
  assert(numpy.all(numpy.array(dims,int) == numpy.array([1,2,0],int)))
  A_reorient = getAffine_orientation(dims,flip)
  A_assert = numpy.array([
    [0, 0, 1, 0],
    [-1, 0, 0, 0],
    [0, -1, 0, 0],
    [0, 0, 0, 1]
  ],float)
  assert(numpy.all(A_reorient == A_assert))

  origin_mm_RAS = parseOriginCode('center','RAS')
  assert(numpy.all(numpy.array(origin_mm_RAS,float)==numpy.array([456*0.025/2, 528*0.025/2, 320*0.025/2],float)))
  A_translate = getAffine_origin(origin_mm_RAS)

  A = toAllen_mm_RAS_center('um','PIR','corner')

  A_allen2sba = numpy.array([
    [ 0.   ,  0.   ,  0.025, -5.7  ],
    [-0.025, -0.   , -0.   ,  5.35 ],
    [-0.   , -0.025, -0.   ,  5.15 ],
    [ 0.   ,  0.   ,  0.   ,  1.   ]
  ])
  print(A)

def convertAllenSpace(fromUnitOrientationOrigin=['um(25)','PIR','corner'],toUnitOrientationOrigin=['mm','RAS','ac']):
  toStandard = toAllen_mm_RAS_center(*fromUnitOrientationOrigin)
  toTarget = toAllen_mm_RAS_center(*toUnitOrientationOrigin)
  return numpy.linalg.inv(toTarget) @ toStandard

def fast_pts2volume(points,dims, values = None):
    if values is not None:
        new_vol = np.zeros(dims, dtype = np.float64)
    else:
        new_vol = np.zeros(dims, dtype = np.uint8)
    points = np.round(points).astype(int)
    a = points[:,0]>=dims[0]; b = points[:,1]>=dims[1]; c = points[:,2]>=dims[2]
    flag_pts = np.logical_or(a,np.logical_or(b,c))
    points = points[~flag_pts]
    (x,y,z) = points[:,0:3].T
    if values is not None:
        values = values[~flag_pts]
        for idx in range(len(values)):
            new_vol[x[idx],y[idx],z[idx]] = values[idx]
    else:
        new_vol[x,y,z] = 1
    
    return new_vol

def fast_volume2pts(vol):
    nzero_pts = np.array(np.nonzero(vol))
    new_pts = nzero_pts.T
    return new_pts

def Make_a_mask(annotation, area, acr2id, hemisphere = 'both'):

    def has_numbers(inputString):
        return any(char.isdigit() for char in inputString)

    if area in acr2id.keys():
        area_mask = np.array(np.where(annotation==acr2id[area])).T
    elif area not in acr2id.keys() and has_numbers(area) is False:  # maybe its just a cortical ancestor
        layers = ['2/3','4','5','6a','6b']
        area_mask = []
        for num in layers:
            tmp = np.array(np.where(annotation==acr2id[area+num])).T
            area_mask.extend(tmp)
        area_mask = np.array(area_mask)
        if len(area_mask) == 0:
            print('Give a legit area')
            return -1
    else:
        print('Give a legit area')
        return -1

    r_midpoint = int(np.round(annotation.shape[2]/2))-1
    if hemisphere == 'left':
        area_mask = area_mask[area_mask[:,2] < r_midpoint]
    elif hemisphere == 'right':
        area_mask = area_mask[area_mask[:,2] >= r_midpoint]

    return area_mask


def swc2volume(infile, res, in_directory = './', intype = 'axon', mode = 'local', annotation = None):

    # infile: the swc file that you are going to load. Give the full directory.
    # mode: can be 'local' (Clasca neurons), 'mouselight' or 'braintell'
    # res: resolution (can be a number for which you have the corresponding allen annotation)
    # intype: choose between 'axon', 'dendrite', or 'both'
    # in_directory: the directory where you have your annotation file

    try:
        with open(infile) as fp:
            neuron = json.load(fp)
    except:
        with gzip.open(infile, 'rb') as fp:
            file_content = fp.read()
            neuron = json.loads(file_content)

    if annotation is None:
        annotation,b = nrrd.read(os.path.join(in_directory,'annotation_{}.nrrd'.format(res)))

    a, Q_sba2allen, c = Affine_transform(neuron, mode = mode,\
                                         out_orientation = ['um({})'.format(res),'PIR','corner'])

    points = numpy.array(neuron['treePoints']['data'])
    lines = numpy.array(neuron['treeLines']['data'])

    VolumeMat = np.zeros(np.shape(annotation)) #(annotation>0).astype(numpy.uint8)
    if intype == 'axon':
        index = [2]
    elif intype == 'dendrite':
        index = [3]
    elif intype == 'both':
        index = [2,3]

    for lineId,line in enumerate(lines):
        pointId = line[1]
        for pointId in range(line[1],line[1]+line[2]):
            coord_RAS_mm_ac = points[pointId][0:3]
            coord_allen = Q_sba2allen[0:3,0:3] @ coord_RAS_mm_ac + Q_sba2allen[0:3,3]
            x = int(np.round(coord_allen[0])); y = int(np.round(coord_allen[1])); z= int(np.round(coord_allen[2]))
            if x >= np.shape(VolumeMat)[0] or y >= np.shape(VolumeMat)[1] or z >= np.shape(VolumeMat)[2]:
                continue
            if line[0] in index:
                VolumeMat[x,y,z] = 1

    new_vol = np.zeros(annotation.shape, dtype = int)

    return VolumeMat

def GetNeuronAnnotation(neuron, neuron_id, structures, annotation25 = None, id2acr = None, out_orientation = ['um(25)','PIR','corner'], \
                         inmode = 'local', dataFolder = '/cortexdisk/data2/NeuronsReunited/full_transformation_surf/data/allenAtlas'):

    if annotation25 is None or id2acr is None:
        res = int(out_orientation[0].split('(')[1].split(')')[0])
        annotation25,allenMeta,acr2id,id2acr,ancestorsById,avg_template25 = getAtlasData( dataFolder, resolution = res)

    points = numpy.array(neuron['treePoints']['data'])
    lines = numpy.array(neuron['treeLines']['data'])

    id2name = {idee:structures['name'][idx] for idx,idee in enumerate(structures['id'])}

    new_neuron, Q_neuron2allen,in_orientation = Affine_transform(neuron, mode = inmode,\
                                                     out_orientation = out_orientation)
    new_neuron, Q_neuron2mm,in_orientation = Affine_transform(neuron, mode = inmode,\
                                                     out_orientation = ['mm','RAS','ac'])
    neuron_cls = NeuronMorphology(neuron)
    somaLineIdx = neuron_cls.getSomaLineIdx()
    distances,branchingOrders = neuron_cls.getPointStatistics()
    axonal_terminal_lines = neuron_cls.get_axonal_terminals()

    region_stats = OrderedDict()
    neuronDict = OrderedDict([('soma',[]),('dendrite',[]),('axon',[]),('allenInformation',[]),('idString', neuron_id)])
    axon_cnt = 0; dend_cnt = 0
    found_id_list = []
    for lineId,line in enumerate(lines):
        lineType,firstPoint,numPoints,prevLineId,negOffset = line
        if lineType == 2:
            prevPoint_mm = None
            if prevLineId:
                prevLine = lines[line[3]]
                prevPoint = points[prevLine[1]+prevLine[2]-1-line[4]]
                prevPoint_mm = Q_neuron2mm[0:3,0:3] @ prevPoint[0:3] + Q_neuron2mm[0:3,3]
            for pointId in range(line[1],line[1]+line[2]):
                point = points[pointId]
                coord_allen = Q_neuron2allen[0:3,0:3] @ point[0:3] + Q_neuron2allen[0:3,3]
                x = int(np.round(coord_allen[0])); y = int(np.round(coord_allen[1])); z= int(np.round(coord_allen[2]))
                if x >= np.shape(annotation25)[0] or y >= np.shape(annotation25)[1] or z >= np.shape(annotation25)[2]:
                    #print('Out of bounds {}'.format(coord_allen))
                    continue
                found_id = annotation25[x,y,z]
                if found_id not in id2acr:
                    print('Unknown id {}'.format(id))
                    continue
                if found_id == 0: id2acr[found_id] = 'background'; id2name[str(found_id)] = 'background'
                point_mm = Q_neuron2mm[0:3,0:3] @ point[0:3] + Q_neuron2mm[0:3,3]
                if lineId in axonal_terminal_lines and pointId == line[1]+line[2]-1: # position of an axonal terminal
                    isterminal = True
                else:
                    isterminal = False
                if prevPoint_mm is not None:
                    allenAcr = id2acr[found_id]
                    if allenAcr not in region_stats.keys():
                        region_stats[allenAcr] = 0
                    s = np.linalg.norm(point_mm - prevPoint_mm)
                    region_stats[allenAcr] += s
                    axon_cnt+=1
                    neuronDict['axon'].append({'x':points[pointId][0],'y':points[pointId][1],'z':points[pointId][2],\
                                            'allenId': found_id,'allenAcr':id2acr[found_id],\
                                            'allenName':id2name[str(found_id)],'parentNumber': prevLineId,\
                                            'axonal length': s, 'sampleNumber': axon_cnt, 'structureIdentifier': 2,\
                                            'radius':1, 'terminal': isterminal, 'distance from soma': distances[pointId]})
                    if found_id not in found_id_list:
                        found_id_list.append(found_id)
                        neuronDict['allenInformation'].append({'allenId': found_id, 'name': id2name[str(found_id)],\
                                                       'acronym':id2acr[found_id]})
                prevPoint_mm = point_mm

        else:
            for pointId in range(line[1],line[1]+line[2]):
                point = points[pointId]
                coord_allen = Q_neuron2allen[0:3,0:3] @ point[0:3] + Q_neuron2allen[0:3,3]
                x = int(np.round(coord_allen[0])); y = int(np.round(coord_allen[1])); z= int(np.round(coord_allen[2]))
                if x >= np.shape(annotation25)[0] or y >= np.shape(annotation25)[1] or z >= np.shape(annotation25)[2]:
                    #print('Out of bounds {}'.format(coord_allen))
                    continue
                found_id = annotation25[x,y,z]
                if found_id not in id2acr:
                    print('Unknown id {}'.format(id))
                    continue
                elif found_id == 0: id2acr[found_id] = 'background'; id2name[str(found_id)] = 'background'
                point_mm = Q_neuron2mm[0:3,0:3] @ point[0:3] + Q_neuron2mm[0:3,3]

                if lineType == 1:
                    neuronDict['soma'] = {'x':points[pointId][0],'y':points[pointId][1],'z':points[pointId][2],\
                                                'allenId':found_id,'allenAcr':id2acr[found_id],\
                                                'allenName':id2name[str(found_id)],'parentNumber': prevLineId}
                elif lineType == 3:
                    dend_cnt +=1
                    neuronDict['dendrite'].append({'x':points[pointId][0],'y':points[pointId][1],'z':points[pointId][2],\
                                                  'allenId':found_id,'allenAcr':id2acr[found_id],\
                                                  'allenName':id2name[str(found_id)],'parentNumber': prevLineId,
                                                  'sampleNumber':dend_cnt,'structureIdentifier':3,\
                                                  'radius':0.5})
                if found_id not in found_id_list:
                        found_id_list.append(found_id)
                        neuronDict['allenInformation'].append({'allenId': found_id, 'name': id2name[str(found_id)],\
                                               'acronym':id2acr[found_id]})

    return neuronDict,region_stats

def Affine_transform(neuron, mode = 'local', out_orientation = ['um','RAS','ac']):

    # make it a numpy array in order for the transformation to be possible
    # neuron: your input morphology, in a python-dictionary format
    # modes =  'local' (Clasca neurons), 'mouselight', 'braintell'
    # out_orientation = vectors comprised of three fields (see explanation in the first lines for the notations of the three fields)
    # 1) unitCode
    # 2) orientationCode
    # 3) originCode
    # This code automatically detectes which orientation system the morphology uses,
    # based on the mode that you give us input, so ensure that you input the term
    # for the correct repository
    # The outputs are: the transformed morphology, the affine transformation matrix, and the orientation of the input morphology

    in_neuron = copy.deepcopy(neuron)
    in_neuron['treePoints']['data'] = np.array(in_neuron['treePoints']['data'])
    points = in_neuron['treePoints']['data']
    scale,res = MicronChecker(in_neuron)
    if mode == 'mouselight':
        orient = 'LIP'
        scale = scale #'um'
        position = 'corner'
    elif mode == 'braintell':
        orient = 'PIR'
        scale = scale #'um'
        position = 'corner'
    elif mode == 'local':
        orient = 'RAS'
        scale = scale #'mm'
        position = 'ac'

    in_orientation = ['{}({})'.format(scale,res),orient,position]
    Q_neuron2allen = convertAllenSpace(fromUnitOrientationOrigin = in_orientation,\
                                            toUnitOrientationOrigin = out_orientation)
    in_neuron['treePoints']['data'][:,0:3] = np.matmul(Q_neuron2allen[0:3,0:3],points[:,0:3].T).T +  Q_neuron2allen[0:3,3]
    in_neuron['treePoints']['data'] = in_neuron['treePoints']['data'].tolist()

    return in_neuron,Q_neuron2allen,in_orientation




def MicronChecker(in_neuron, orient = 'PIR'):
    # We can improve it based on the above functions
    # MicronChecker checks the resolution of your input morphology (in_neuron)
    # ensure that you have set the correct orientation (orient field, for the notation see explanation above)
    somaCoord = []
    for line in in_neuron['treeLines']['data']:
          if line[0] == 1:
            # soma
            somaCoord = numpy.array(in_neuron['treePoints']['data'][line[1]],numpy.float)
            somaCoord[3] = 1.0
            break;

    if somaCoord[0] < 10 and somaCoord[1] < 10 and somaCoord[2] < 10:
        res = 'mm'
        unit = 1
    else:
        res = 'um'
        if orient == 'PIR':
            P,I,R = (528, 320, 456)
            unit = 1 if (somaCoord[0]>P or somaCoord[1]>I or somaCoord[2]>R) else 25
        elif orient == 'LIP':
            L,I,P = (450, 320, 528)
            unit = 1 if (somaCoord[0]>L or somaCoord[1]>I or somaCoord[2]>P) else 25
    return res,unit
