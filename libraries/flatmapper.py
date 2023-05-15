# import os
# import re
# import nrrd
# import numpy as np
# import nibabel
# from PIL import Image
# import tempfile
# from lxml import etree
# from matplotlib import pyplot as plt
from cfg import *
sys.path.append('atlasoverlay')
import atlasoverlay as ao


# HELPER ROUTINES

# Compute array of unique and contrasting colors
def contrastingColors(numColors):
    base = int(np.ceil(numColors**(1/3)))
    if base>256:
        raise RuntimeError( 'base ({}) must be 256 or below'.format(base) )
    base2 = base*base
    scaleup = 256 // base
    rgb_dtype = np.dtype([('R', 'u1'), ('G', 'u1'), ('B', 'u1')])
    index2rgb = np.zeros(numColors,rgb_dtype)
    for index in range(0,numColors):
        index2rgb[index] = (
          (index % base) * scaleup,
          ((index % base2) // base) * scaleup,
          (index // base2) * scaleup
        )
    return index2rgb

# Apply a smoothing filter to each color of a label image, and for each
# pixel keep the color with the maximum intensity after smoothing.
#
# img is a 2d array of label-ids
# filter contains 1-d array of weights, to be applied in both dimensions.
# ignore is a list of label-ids that should be left untouched.
def multiLabelSmooth(img,filter,ignore=[]):
  r = len(filter)//2
  sz = img.shape
  tp = img.dtype

  maximg = np.zeros(sz,float)
  newimg = np.zeros(sz,tp)
  regions = np.unique(img)
  for g in regions:
    if g in ignore: continue
    # np.dot: For N dimensions it is a sum product over the last axis of a and the second-to-last of b
    filteredslice0 = (img==g).astype(float)
    # filter over i-dimension
    filteredslice = np.zeros_like(filteredslice0)
    for i,coeff in enumerate(filter):
      d = i-r
      if d<0:
        filteredslice[-d:,:] += coeff*filteredslice0[:d,:]
      elif d>0:
        filteredslice[:-d,:] += coeff*filteredslice0[d:,:]
      else:
        filteredslice += coeff*filteredslice0
    filteredslice0 = filteredslice
    # filter over k-dimension
    filteredslice = np.zeros_like(filteredslice0)
    for k,coeff in enumerate(filter):
      d = k-r
      if d<0:
        filteredslice[:,-d:] += coeff*filteredslice0[:,:d]
      elif d>0:
        filteredslice[:,:-d] += coeff*filteredslice0[:,d:]
      else:
        filteredslice += coeff*filteredslice0
    maximg = np.maximum(maximg,filteredslice)
    newimg[np.logical_and(maximg==filteredslice,maximg>0)] = g
  return newimg

# expects SVG in which areas are grouped, and the group-id is formatted as "#rrggbb,id,acr"
def bestLabelPlacement(svgString,bothHemispheres=False):
    from svgpath2mpl import parse_path
    from shapely.geometry import Polygon
    from polylabel_rb import polylabel

    # Use lxml to read the paths from svg
    tree = etree.fromstring(svgString)
    # place labels as far as possible on the left
    svgElems = tree.xpath('//*[name()="svg"]')
    if bothHemispheres:
        xMid = int(svgElems[0].attrib['width'])//2
    else:
        xMid = int(svgElems[0].attrib['width'])

    rgb2id = {}
    largestLeftCentroid = {}
    largestRightCentroid = {}
    gElems = tree.xpath('//*[name()="g"]')
    for elem in gElems:
        attrs = elem.attrib
        if 'id' in attrs:
            m = re.search(r'(#[\da-fA-F]+),(\d*),(.*)',attrs['id'])
            if m:
                rgb = m.group(1).lower()
                id = int(m.group(2))
                rgb2id[rgb] = id
                for ch in elem.iterchildren():
                    path = parse_path(ch.get('d'))
                    vertices = path._vertices
                    poly = Polygon(vertices).buffer(5)
                    centroid,radius = polylabel(poly)
                    if centroid.x<=xMid:
                        if id not in largestLeftCentroid or radius>largestLeftCentroid[id][1]:
                            largestLeftCentroid[id] = ((centroid.x,centroid.y),radius)
                    else:
                        if id not in largestRightCentroid or radius>largestRightCentroid[id][1]:
                            largestRightCentroid[id] = ((centroid.x,centroid.y),radius)

    # Place labels as far as possible in the left hemisphere
    largestCentroid = {}
    for rgb,id in rgb2id.items():
        cL = largestLeftCentroid[id] if id in largestLeftCentroid else None
        cR = largestRightCentroid[id] if id in largestRightCentroid else None
        if cL and cR:
            largestCentroid[id] = cL if cL[1]>0.8*cR[1] else cR
        elif cL or cR:
            largestCentroid[id] = cL if cL else cR

    return largestCentroid

# CLASS FLATMAPPER
class FlatMapper:
    projectionTypes = ['back','flatmap_dorsal','side','bottom','front','medial','flatmap_butterfly','rotated','top']
    shape10 = (1320, 800, 1140)
    shape10_lh = (1320, 800, 570)

    def __init__(self,pathToStreamlines):
        self.pathToStreamlines = pathToStreamlines
        self.projectionType = None

    def init(self,projectionType):
        if projectionType is not self.projectionType:
            self.done()
        if self.projectionType is None:
            self.view_lookup_lh,hdr = nrrd.read('{}/view_lookup_sorted1({},lh).nrrd'.format(self.pathToStreamlines,projectionType))
            hdr = nrrd.read_header('{}/{}.nrrd'.format(self.pathToStreamlines,projectionType))
            self.resultShape = hdr['sizes']
            self.cortex_path_lookup_lh,hdr = nrrd.read('{}/cortex_path_lookup_lh.nrrd'.format(self.pathToStreamlines))
            self.cortex_mask_lh,hdr = nrrd.read('{}/cortex_mask_lh.nrrd'.format(self.pathToStreamlines))
            self.paths_lh,hdr = nrrd.read('{}/surface_paths_10(lh).nrrd'.format(self.pathToStreamlines))
            self.projectionType = projectionType

    def done(self):
        self.view_lookup_lh = None
        self.cortex_path_lookup_lh = None
        self.paths_lh = None
        self.projectionType = None

    # (debug) show which part of all surface voxels are actually part of this projection
    def debugProjectedSurface(self,saveAs=None,doPlot=False):
        path_lookup_lh,hdr = nrrd.read('{}/surface_path_lookup_10(lh).nrrd'.format(self.pathToStreamlines))
        path_lookup_lh[path_lookup_lh>-1] = 127
        path_lookup_lh[path_lookup_lh==-1] = 0
        path_lookup_lh = path_lookup_lh.astype(np.uint8)
        for sv in self.view_lookup_lh[:,1]:
            path_lookup_lh.flat[sv] = 255
        if saveAs is None:
            saveAs = '/tmp/projectedsurface.nrrd'
        if saveAs:
            nrrd.write(saveAs,path_lookup_lh)
        if doPlot:
            self.plotVolume(path_lookup_lh)

    # (debug) show which part of all surface voxels are actually part of this projection
    def debugProjectedQuadrant(self,saveAs=None,doPlot=False,gridSize=2):
        quadVolume = np.zeros(self.cortex_path_lookup_lh.shape,np.uint8)
        for px,sv in self.view_lookup_lh:
            xy = np.unravel_index(px,self.resultShape)
            quadrant = 1+gridSize*((xy[0]*gridSize)//self.resultShape[0]) + (xy[1]*gridSize)//self.resultShape[1]
            quadVolume.flat[sv] = quadrant
        if saveAs is None:
            saveAs = '/tmp/quadvolume.nrrd'
        if saveAs:
            nrrd.write(saveAs,quadVolume)
        if doPlot:
            self.plotVolume(quadVolume)


    # (debug) plot cortex path lookup to see if all of cortex is covered
    def debugCortexPathLookup(self):
        self.plotVolume(self.cortex_path_lookup_lh)

    # (debug) inspect volume by plotting three orthogonal slices
    def plotVolume(self,data,affine=np.eye(4)):
        shape = data.shape
        wLR = shape[0]*affine[0,0]
        wPA = shape[1]*affine[1,1]
        wIS = shape[2]*affine[2,2]
        fig,ax = plt.subplots(1,3,figsize=(15,15),gridspec_kw={'width_ratios': [wLR,wPA,wLR]})
        print(ax.shape)
        ax = ax.reshape([1,ax.size])
        print('SHAPE',ax.shape)

        sliceLR = shape[0]//5*3+3
        slicePA = shape[1]//5*2+3
        sliceIS = shape[2]//10*7
        ax[0,0].imshow(data[:,slicePA,::-1].T,cmap=plt.get_cmap('gray'),aspect=1)
        ax[0,1].imshow(data[sliceLR,::-1,::-1].T,cmap=plt.get_cmap('gray'),aspect=2/5)
        ax[0,2].imshow(data[:,::-1,sliceIS].T,cmap=plt.get_cmap('gray'),aspect=5/2)

        for r in ax:
            r[0].plot([0,shape[0]],[shape[2]-1-sliceIS,shape[2]-1-sliceIS],'b:')
            r[0].plot([sliceLR,sliceLR],[0,shape[2]],'g:')
            r[0].plot([0,shape[0],shape[0],0,0],[0,0,shape[2],shape[2],0],'r')
            r[1].plot([0,shape[1]],[shape[2]-1-sliceIS,shape[2]-1-sliceIS],'b:')
            r[1].plot([shape[1]-1-slicePA,shape[1]-1-slicePA],[0,shape[2]],'r:')
            r[1].plot([0,shape[1],shape[1],0,0],[0,0,shape[2],shape[2],0],'g')
            r[2].plot([0,shape[0]],[shape[1]-1-slicePA,shape[1]-1-slicePA],'r:')
            r[2].plot([sliceLR,sliceLR],[0,shape[1]],'g:')
            r[2].plot([0,shape[0],shape[0],0,0],[0,0,shape[1],shape[1],0],'b')
            for c in r:
                c.axis('off')

    # minimumThickness in voxels along streamline before applying mask
    def projectVolume(self, dataVolume,projectionType='flatmap_dorsal',aggregateFunction=np.mean, minimumThickness=None, maskVolume=None, savePng=None,saveNifti=None):
        # labelVolume=None,labelSelection=None,
        if not np.all(dataVolume.shape == self.shape10_lh):
            if np.all(dataVolume.shape == self.shape10):
                print('Using only left hemisphere of dataVolume');
                dataVolume = dataVolume[:,:,:self.shape10_lh[2]//2]
            else:
                raise RuntimeError('Data volume has incorrect shape: {} instead of {} or {}'.format(dataVolume.shape,self.shape10_lh,self.shape10))
        if maskVolume is not None:
            if not np.all(maskVolume.shape == self.shape10_lh):
                if np.all(maskVolume.shape == self.shape10):
                    print('Using only left hemisphere of maskVolume');
                    maskVolume = maskVolume[:,:,:self.shape10_lh[2]//2]
                else:
                    raise RuntimeError('Label volume has incorrect shape: {} instead of {} or {}'.format(maskVolume.shape,self.shape10_lh,self.shape10))

        def apply_along_path(p):
            path = self.paths_lh[p,:]
            path = path[path>0]
            if minimumThickness:
                if len(path)<minimumThickness:
                    return 0
            #if labelVolume is not None:
            #    path = [ path[i] for i,label in enumerate(labelVolume.flat[path]) if label in labelSelection ]
            if maskVolume is not None:
                path = [ p for p in path if maskVolume.flat[p] ]

            # support dual hemisphere volumes
            """
            ijk = np.unravel_index(path,(1320, 800, 1140))
            rh = ijk[2]>=dataVolume.shape[2]
            ijk[2][rh] = dataVolume.shape[2]-1-ijk[2][rh]
            path = np.ravel_multi_index(ijk,dataVolume.shape)
            """
            if len(path):
                values = dataVolume.flat[path]
                return aggregateFunction(values)
            else:
                return 0

        if not callable(aggregateFunction):
            raise ValueError('Aggregate function must be a callable function, not {}'.format(aggregateFunction))

        # initialize output
        self.init(projectionType)
        result = np.zeros(self.resultShape, dtype=float)
        applyAggregate = np.vectorize(apply_along_path)
        resultIndices = self.view_lookup_lh[:,0]
        voxelIndices = self.view_lookup_lh[:,1]
        pathIndices = self.cortex_path_lookup_lh.flat[voxelIndices]
        result.flat[resultIndices] = applyAggregate(pathIndices)
        result = result.T

        self.done()

        if savePng:
            mx = result.max()
            if mx<255.9999: mx = 255.9999
            im = Image.fromarray((result*255.9999/mx).astype(np.uint8))
            im.save(savePng)
        if saveNifti:
            # for inspection with ITK-snap
            nii = nibabel.Nifti1Image(result,np.eye(4))
            nibabel.save(nii,saveNifti)
        return result

    # alias
    project = projectVolume

    def findNearestSurfaceVoxel(self,vx):
        # surfaceVoxel vx is not part of this projection, try any of its neighbors
        ijk = np.array(np.unravel_index(vx,self.shape10_lh),int)
        sequence = [0,-1,1,-2,2,-3,3]
        for radius in [1,2]:
            for di in sequence[0:2*radius+1]:
                for dj in sequence[0:2*radius+1]:
                    for dk in sequence[0:2*radius+1]:
                        ijkNb = ijk+(di,dj,dk)
                        if np.any(ijkNb>=self.shape10_lh) or np.any(ijkNb<0):
                            continue
                        nb = np.ravel_multi_index(ijkNb,self.shape10_lh)
                        # get surface voxel for this neighbor
                        pathIndex = self.cortex_path_lookup_lh.flat[nb]
                        if pathIndex > -1:
                            nb = self.paths_lh[pathIndex,0]
                            matches = np.flatnonzero(self.view_lookup_lh[:,1]==nb)
                            if len(matches):
                                return matches[0]
        return -1


    def projectPoints(self, points,projectionType='flatmap_dorsal',debug=False):
        self.init(projectionType)
        if not isinstance(points,np.ndarray):
            points = np.array(points,np.uint32)
        rh = points[:,2] >= self.shape10_lh[2]
        pointsL = points.copy()
        pointsL[rh,2] = self.shape10[2]-1-pointsL[rh,2]
        voxelIndices = np.ravel_multi_index((pointsL[:,0],pointsL[:,1],pointsL[:,2]), self.shape10_lh)
        pathIndices = self.cortex_path_lookup_lh.flat[voxelIndices]
        xy = np.zeros((points.shape[0],2),np.int32)-1
        hasMatch = []
        validMatches = []
        for i,pathIndex in enumerate(pathIndices):
            if pathIndex>-1:
                sv = self.paths_lh[pathIndex,0]
                # time consuming step...
                matches = np.flatnonzero(self.view_lookup_lh[:,1]==sv)
                if len(matches):
                    match = matches[0]
                else:
                    match = self.findNearestSurfaceVoxel(sv)
                if match > -1:
                    hasMatch.append(i)
                    validMatches.append( match )
                else:
                    print('ERROR: no matching surface voxel found for point ',points[i,:])
        xy[hasMatch,1],xy[hasMatch,0] = np.unravel_index(self.view_lookup_lh[validMatches,0],self.resultShape)
        return xy

    def createAnnotationSvg(self,projectedAnnotation, id2acr, id2rgb=None, acr2full=None, strokeWidth=None, smoothness=None, mlFilter=None, saveNifti=None):
        if strokeWidth is None:
            strokeWidth = projectedAnnotation.shape[0]/300
        if smoothness is None:
            smoothness = projectedAnnotation.shape[0]/150

        # Convert array of id-values to rgb array that can be saved as an image
        def imagify(idVolume,id2rgb=None):
            unique = np.unique(idVolume).astype(np.uint32)
            unique.sort()
            numColors = len(unique)
            indexVolume = np.zeros(idVolume.shape,np.uint8 if numColors<=256 else np.uint16 if numColors<=65536 else np.uint32)
            for index,id in enumerate(unique):
                indexVolume[idVolume==id] = index

            if id2rgb is None:
                index2rgb = contrastingColors(numColors)
            else:
              rgb_dtype = np.dtype([('R', 'u1'), ('G', 'u1'), ('B', 'u1')])
              index2rgb = np.zeros(numColors,rgb_dtype)
              for id,rgb in id2rgb.items():
                  index = np.where(unique==int(id))
                  if len(index):
                      index2rgb[index[0]] = tuple(int(rgb[i:i+2],16) for i in (1, 3, 5))
            rgb2id = { "#{:02x}{:02x}{:02x}".format(*rgb):unique[int(index)] for index,rgb in enumerate(index2rgb) }

            rgbVolume = index2rgb[indexVolume]
            return rgbVolume,rgb2id

        idImage  = projectedAnnotation
        if mlFilter is not None:
            idImage = multiLabelSmooth(idImage,mlFilter)
            if saveNifti:
                nii = nibabel.Nifti1Image(idImage,np.eye(4))
                nibabel.save(nii,saveNifti.replace('.nii','_smooth({}).nii'.format(','.join([str(v) for v in mlFilter]))))

        backgroundId = idImage[0,0]
        rgbImage,rgb2id = imagify(idImage,id2rgb)
        with tempfile.TemporaryDirectory() as tmpdir:
            im = Image.fromarray(rgbImage.view(np.uint8).reshape(rgbImage.shape+(3,)))
            imageFile = os.path.join(tmpdir,'image.png')
            im.save(imageFile)
            svgString = ao.getSvgContours(imageFile, strokeColor = 'auto', strokeWidth=strokeWidth, smoothness=smoothness, rgb2id=rgb2id,id2acr=id2acr,acr2full=acr2full)

        labelCoords = bestLabelPlacement(svgString)
        fontSize_px = idImage.shape[0]/60
        s = ['<g id="area_labels" style="fill:#000; text-anchor: middle; dominant-baseline: middle; font-size:{}px; font-family: sans-serif">'.format(fontSize_px)]
        for id,coord in labelCoords.items():
            if id != backgroundId:
                s.append('<g><text stroke-width="{}" stroke="#666" x="{}" y="{}">{}</text>'.format(fontSize_px/10,coord[0][0],coord[0][1],id2acr[str(id)]))
                s.append('<text x="{}" y="{}">{}</text></g>'.format(coord[0][0],coord[0][1],id2acr[str(id)]))
        s.append('</g>')
        svgString = svgString[:-6]+"\n".join(s)+svgString[-6::]

        return svgString


    # AGGREGATE FUNCTIONS
    @staticmethod
    def corticalAreaFunc(index2id,ancestorsById,allowedParentIds):
        # Find out which areas have children
        hasChildren = set()
        for id,ancestors in ancestorsById.items():
            if len(ancestors) > 1:
                parent = ancestors[1]
                hasChildren.add(parent)

        def AF(values):
            nonzero = [index2id[i] for i in values if index2id[i]>0]
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

    @staticmethod
    def layerMapFunc(idsByLayer,index2id):
        def AF(values):
            nonzero = [index2id[i] for i in values if index2id[i]>0]
            if len(nonzero):
                hasLayer = [0,0,0,0,0,0]
                for id in nonzero:
                    if id in idsByLayer['Layer1']:
                        hasLayer[0] = 1
                    elif id in idsByLayer['Layer2_3']:
                        hasLayer[1] = 1
                    elif id in idsByLayer['Layer4']:
                        hasLayer[2] = 1
                    elif id in idsByLayer['Layer5']:
                        hasLayer[3] = 1
                    elif id in idsByLayer['Layer6a']:
                        hasLayer[4] = 1
                    elif id in idsByLayer['Layer6b']:
                        hasLayer[5] = 1
                return hasLayer[0]+2*hasLayer[1]+4*hasLayer[2]+8*hasLayer[3]+16*hasLayer[4]+32*hasLayer[5]
            return 0
        return AF

    @staticmethod
    def countLayersFunc(idsByLayer,index2id):
        def AF(values):
            nonzero = [index2id[i] for i in values if index2id[i]>0]
            if len(nonzero):
                hasLayer = np.zeros((len(idsByLayer),),np.uint8)
                for id in nonzero:
                    if id in idsByLayer['Layer1']:
                        hasLayer[0] = 1
                    elif id in idsByLayer['Layer2_3']:
                        hasLayer[1] = 1
                    elif id in idsByLayer['Layer4']:
                        hasLayer[2] = 1
                    elif id in idsByLayer['Layer5']:
                        hasLayer[3] = 1
                    elif id in idsByLayer['Layer6a']:
                        hasLayer[4] = 1
                    elif id in idsByLayer['Layer6b']:
                        hasLayer[5] = 1
                return hasLayer.sum()
            return 0
        return AF

    # to find out whether a streamline passes any voxels that are not cortical layers
    @staticmethod
    def nonlayerMapFunc(idsByLayer,index2id):
        id2index = { id:index for index,id in enumerate(index2id) }
        def AF(values):
            nonzero = [index2id[i] for i in values if index2id[i]>0]
            if len(nonzero):
                for id in nonzero:
                    anyLayer = False
                    for key in idsByLayer:
                        if id in idsByLayer[key]:
                            anyLayer = True
                            break
                    if not anyLayer:
                        return id2index[id]
            return 0
        return AF
