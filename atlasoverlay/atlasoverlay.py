import svgutils_transform as svgu

import sys
sys.path.append('../libraries')
from cfg import *

hex2rgb = lambda x : tuple(int(x.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))

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

# Convert array of id-values to rgb array that can be saved as an image
def imagify(idVolume,id2rgb=None):
  unique = np.unique(idVolume)
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
      index = np.where(unique==id)
      if len(index):
        index2rgb[index[0]] = tuple(int(rgb[i:i+2],16) for i in (1, 3, 5))

  rgb2id = { "#{:02x}{:02x}{:02x}".format(*rgb):unique[index] for index,rgb in enumerate(index2rgb) }
  rgbVolume = index2rgb[indexVolume]
  return rgbVolume,rgb2id

def flat2png(flat_infile, png_outfile = 'flat.png'):

    with open(flat_infile,'rb') as fp:
        buffer = fp.read()
    nDims = int(buffer[0])
    shape = np.frombuffer(buffer, dtype=np.dtype('>i4'), offset=1, count=2)
    data = np.frombuffer(buffer, dtype=np.dtype('>i2'), offset=9)
    data = data.reshape(shape[::-1])
    #print(nDims,shape,data.shape,data.min(),data.max())

    rgbdata,rgb2id = imagify(data)
    rgbdata3 = rgbdata.view(np.uint8).reshape(rgbdata.shape+(3,))
    #plt.imshow(rgbdata3)
    flat = Image.fromarray(rgbdata3)
    flat.save(png_outfile)

def ProduceSvg(newimg, in_dir, labels, labels_extra = None, savefile = None,
               in_color = '#000', overlay_dir = '/cortexdisk/data2/NestorRembrandtCollab/Code Repositories/atlasoverlay/'):

    cur_dir = os.getcwd()
    newimg_cpy = deepcopy(newimg)

    figSize_inch = (8,6) # width, height in inches
    dpi = 100
    fig = plt.figure(figsize=figSize_inch, dpi=dpi)

    ax = plt.subplot(1,1,1)

    zero_mask_lab = numpy.all(labels == [0,0,0], axis=-1)
    labels[:,:,0][zero_mask_lab] = 255
    labels[:,:,1][zero_mask_lab] = 255
    labels[:,:,2][zero_mask_lab] = 255
    if labels_extra is not None:
        pos = ax.imshow(labels_extra,plt.get_cmap('gray'))
    else:
        pos = ax.imshow(labels,plt.get_cmap('gray'))

    cmap_mask  = plt.get_cmap('gray')

    nzero_mask = newimg > 0
    zero_mask_pd = newimg == 0

    newimg_cpy[nzero_mask] += 10000
    newimg_rgb = cmap_mask(newimg_cpy, bytes = True)

    if in_color != '#000':
        in_color_rgb = hex2rgb(in_color)
        newimg_rgb[:,:,0][nzero_mask] = in_color_rgb[0]
        newimg_rgb[:,:,1][nzero_mask] = in_color_rgb[1]
        newimg_rgb[:,:,2][nzero_mask] = in_color_rgb[2]
    else:
        newimg_rgb[:,:,0][nzero_mask] = 1
        newimg_rgb[:,:,1][nzero_mask] = 1
        newimg_rgb[:,:,2][nzero_mask] = 1

    svg = array2svg(newimg_rgb)

    svgFig = matplot2svg(fig)
    atlasOverlay(svgFig, ax, svg, in_path = overlay_dir)
    atlasOverlay(svgFig, ax, labels, in_path = overlay_dir,         # os.path.join(cur_dir, png_file)
                        strokeWidth = 0.3, strokeColor = '#000', contourSmoothness=0.1)
    atlasOverlay(svgFig, ax, svg, in_path = overlay_dir)
    displaySvgFigure(svgFig)
    svg = stringifySvgFigure(svgFig)

    # Save the figure to an svg file
    if savefile is not None:
        savefile = savefile.split('.')[0] + '.svg'
        with open(os.path.join('export', savefile),'wt') as fp:
            fp.write(svg)


def createAnnotationSvg(projectedAnnotation, id2acr, id2rgb=None, acr2full=None, strokeWidth=None, smoothness=None, mlFilter=None, saveNifti=None):
  if strokeWidth is None:
    strokeWidth = projectedAnnotation.shape[0]/300
  if smoothness is None:
    smoothness = projectedAnnotation.shape[0]/150

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
    svgString = getSvgContours(imageFile, strokeColor = 'auto', strokeWidth=strokeWidth, smoothness=smoothness, rgb2id=rgb2id,id2acr=id2acr,acr2full=acr2full)

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

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def image2svg(pngImage,rgbHex):
  if rgbHex[0] == '#': rgbHex = rgbHex[1:]
  r = int(rgbHex[0:2],16)
  g = int(rgbHex[2:4],16)
  b = int(rgbHex[4:6],16)
  im = Image.open(pngImage)
  data = np.array(im)
  rgba_dtype = np.dtype([('R', 'u1'), ('G', 'u1'), ('B', 'u1'), ('A', 'u1')])
  rgbImage = np.zeros(data.shape,dtype=rgba_dtype)
  rgbImage['R'] = r
  rgbImage['G'] = g
  rgbImage['B'] = b
  rgbImage['A'] = 255.9999*np.sqrt(data/256)
  im = Image.fromarray(rgbImage.view(np.uint8).reshape(rgbImage.shape+(4,)))
  pngBytes = io.BytesIO()
  im.save(pngBytes, format='PNG')

  imageSvg = """<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="{}" height="{}"><image width="{}" height="{}" xlink:href="data:image/png;base64,{}"/></svg>""".format(im.width,im.height,im.width,im.height,base64.b64encode(pngBytes.getvalue()).decode('utf-8'))
  return imageSvg

def array2svg(data):
  rgba_dtype = np.dtype([('R', 'u1'), ('G', 'u1'), ('B', 'u1'), ('A', 'u1')])
  rgbImage = np.zeros(data.shape[0:2],dtype=rgba_dtype)
  opacity = np.zeros(data.shape[0:2], np.uint8)
  opacity[data[:,:,0]>0] = 255
  opacity[data[:,:,1]>0] = 255
  opacity[data[:,:,2]>0] = 255
  rgbImage['R'] = data[:,:,0]
  rgbImage['G'] = data[:,:,1]
  rgbImage['B'] = data[:,:,2]
  rgbImage['A'] = opacity
  im = Image.fromarray(rgbImage.view(np.uint8).reshape(rgbImage.shape+(4,)))
  pngBytes = io.BytesIO()
  im.save(pngBytes, format='PNG')

  imageSvg = """<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="{}" height="{}"><image width="{}" height="{}" xlink:href="data:image/png;base64,{}"/></svg>""".format(im.width,im.height,im.width,im.height,base64.b64encode(pngBytes.getvalue()).decode('utf-8'))
  return imageSvg

# img is 2d array of ids
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
def bestLabelPlacement(svgString):
  from svgpath2mpl import parse_path
  from shapely.geometry import Polygon
  from polylabel_rb import polylabel

  # Use lxml to read the paths from svg
  tree = etree.fromstring(svgString)
  # place labels as far as possible on the left
  svgElems = tree.xpath('//*[name()="svg"]')
  xMid = int(svgElems[0].attrib['width'])//2

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


def vectorizeLabelImage(labelImage,smoothness=0,bgColor='auto', in_path = None):
  curveTolerance = 1.0*smoothness;
  lineTolerance = 0.5*smoothness;
  # print('SMOOTHNESS',curveTolerance,lineTolerance)
  with tempfile.TemporaryDirectory() as tmpdir:
    os.makedirs(tmpdir,exist_ok=True)
    svgFile = os.path.join(tmpdir,'labelimage.svg')
    if in_path is None:
        in_path = os.getcwd()
    prog = os.path.abspath(os.path.join(in_path,'mindthegap/bin/mindthegap'))
    cmd = [prog,"-i",labelImage,"-o",svgFile,"-t",str(curveTolerance),"-s",str(lineTolerance)]
    if bgColor != 'auto':
      cmd.extend(('-c',bgColor))
      # print('USING BGCOLOR',bgColor,cmd)
    ans = subprocess.check_output(cmd, shell=False, stderr=subprocess.STDOUT)
    with open(svgFile,'rt') as fp:
      return fp.read()

def getSvgContours(labelImage,strokeColor='auto',strokeWidth=0.5,smoothness=0, rgb2id=None,id2acr=None,acr2full=None, in_path = None):
  parser = etree.XMLParser(remove_blank_text=True)
  if re.fullmatch('.*\.svg',labelImage):
    tree = etree.parse(labelImage,parser)
  else:
    im = Image.open(labelImage)
    im = im.convert('RGB')
    bgColor = "#{:02x}{:02x}{:02x}".format( *im.getpixel((0,0)) )
    svgString = vectorizeLabelImage(labelImage,smoothness,bgColor, in_path)
    tree = etree.fromstring(svgString,parser)
    # remove rectangle elements
    paths = tree.xpath('//*[name()="rect"]')
    for path in paths:
      path.getparent().remove(path)
    # unfill paths, use stroke instead
    paths = tree.xpath('//*[name()="path"]')
    rgbGroups = {}
    for p in paths:
      rgb = p.get('fill')
      if rgb == bgColor:
        p.getparent().remove(p)
      else:
        if rgb in rgbGroups:
          rgbGroups[rgb].append(p)
        else:
          rgbGroups[rgb] = [p]

        p.attrib.pop('fill')
        p.attrib.pop('stroke')
        p.attrib.pop('stroke-width')
    for rgb,paths in rgbGroups.items():
      rgb = rgb.lower()
      id = rgb2id[rgb] if (rgb2id and rgb in rgb2id) else ''
      acr = id2acr[id] if (id2acr and id in id2acr) else ''
      acr = urllib.parse.quote(acr)
      full = acr2full[acr] if (acr2full and acr in acr2full) else ''
      full = urllib.parse.quote(full)
      stroke = rgb if (strokeColor=='auto') else strokeColor
      g = etree.SubElement(tree, "g",{"id":"{},{},{},{}".format(rgb,id,acr,full),"fill":"none","stroke":stroke,"stroke-width":str(strokeWidth)})
      for p in paths:
        p.getparent().remove(p)
        g.append(p)

  svg = etree.tostring(tree).decode('utf-8')
  return svg

def matplot2svg(fig):
  # Create svg container figure
  sz = fig.get_size_inches()
  svgFig = svgu.SVGFigure('{}in'.format(sz[0]), '{}in'.format(sz[1]))
  svgFig.set_size(['{}in'.format(sz[0]), '{}in'.format(sz[1])])
  svg_layer = svgu.from_mpl(fig)
  svgFig.append(svg_layer.getroot())
  plt.close(fig)
  return svgFig

def getSize(svgFig):
  (width,height) = svgFig.get_size()
  if width[-2::] == 'in':
    width = numpy.round(72*float(width[:-2:]))
    height = numpy.round(72*float(height[:-2:]))
  else:
    width = int(width)
    height = int(height)
  return (width,height)

def atlasOverlay(svgFig,mplAxis,warpedAtlasSlice,strokeColor="auto",strokeWidth=0.5,contourSmoothness=1, in_path = None):
  figBox = mplAxis.get_position()
  (width,height) = getSize(svgFig)

  # save warpedAtlasSlice if supplied as ndarray
  if isinstance(warpedAtlasSlice,numpy.ndarray):
    tmpdir = tempfile.gettempdir()
    im = Image.fromarray(warpedAtlasSlice)
    warpedAtlasSlice = os.path.join(tmpdir,'warpedAtlasSlice.png')
    im.save(warpedAtlasSlice)

  # Load warpedAtlasSlice and detect size
  try:
    with Image.open(warpedAtlasSlice) as im:
      svgSize = [im.width,im.height]
    # Use the mind-the-gap algorithm to convert the image to svg contours
    svgString = getSvgContours(warpedAtlasSlice,strokeColor=strokeColor,strokeWidth=strokeWidth,smoothness=contourSmoothness, in_path = in_path)
  except:
    try:
      tree = etree.parse(warpedAtlasSlice)
    except:
      tree = etree.fromstring(warpedAtlasSlice)
    svgElems = tree.xpath('//*[name()="svg"]')
    attrs = svgElems[0].attrib
    svgSize = (int(attrs['width']),int(attrs['height']))
    svgString = etree.tostring(tree).decode('utf-8')

  svg_layer = svgu.fromstring( svgString )

  # Fit the svg contours to the specified figBox
  svg_layer.set_size([str(width),str(height)])
  root_layer = svg_layer.getroot()
  root_layer.moveto(width*figBox.x0+0.25,height*(1.0-figBox.y1)-0.25)
  axSize = [width*(figBox.x1-figBox.x0),height*(figBox.y1-figBox.y0)]
  root_layer.scale_xy(axSize[0]/svgSize[0],axSize[1]/svgSize[1])
  svgFig.append(root_layer)

def displaySvgFigure(svgFig):
  (width,height) = getSize(svgFig)
  svg = svgFig.to_str()
  tree = etree.fromstring(svg)
  paths = tree.xpath('//*[name()="svg"]')
  paths[0].set('viewBox','0 0 {} {}'.format(width,height))
  svg = etree.tostring(tree).decode('utf-8')
  display(SVG(svg))

def stringifySvgFigure(svgFig):
  svg = svgFig.to_str()
  svg = re.sub(r'\s*viewBox="[^"]+"','',svg.decode("utf-8"))
  return svg

def saveSvgFigure(svgFig,outputFile):
  with open(outputFile,'wt') as fp:
    fp.write( stringifySvgFigure(svgFig) )
