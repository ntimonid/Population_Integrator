
from cfg import *
from allensdk.core.reference_space_cache import ReferenceSpaceCache


def getAtlasData(dataFolder='.', resolution = 25):

    try:
        [annotation, allenMeta] = nrrd.read('{}/annotation_{}.nrrd'.format(dataFolder,resolution))
        [avg_template, allenMeta] = nrrd.read('{}/average_template_{}.nrrd'.format(dataFolder,resolution))
    except:
        # ID 1 is the adult mouse structure graph
        reference_space_key = 'annotation/ccf_2017'
        rspc = ReferenceSpaceCache(resolution, reference_space_key, manifest='manifest.json')
        annotation, allenMeta = rspc.get_annotation_volume()

    with open('{}/ancestorsById.json'.format(dataFolder)) as fp:
          ancestorsById = json.load(fp)
    with open('{}/annot_acr2id.json'.format(dataFolder)) as fp: # Latest change 04/10/2022: Introduced new acr2id
          acr2id = json.load(fp)

    try:
        EUAL_annotation, ysk_hdr = nrrd.read('{}/YSK_annotation_{}.nrrd'.format(dataFolder, resolution))
    except:
        print('The EUAL annotation file cannot be found! Either select another directory or another resolution...')
        EUAL_annotation = []

    try:
        projected_annotation, proj_an_hdr = nrrd.read('{}/projectedAnnotation_{}.nrrd'.format(dataFolder, resolution))
    except:
        print('The projection annotation file cannot be found! Either select another directory or another resolution...')
        projected_annotation = []

    id2acr = { id:acr for acr,id in acr2id.items() }
    if 0 not in id2acr:
        acr2id['[background]'] = 0
        id2acr[0] = '[background]'

    return annotation, acr2id, id2acr, ancestorsById, avg_template, EUAL_annotation, projected_annotation


# This function takes as input the list that the parallel for-loop produces, and then reformats it as a dictionary
# Each dictionary key has the name of each registered section image
def reformat_as_dictionary(input_list):
    out_dict = {key.split('\\')[1].split('.')[0] :val for i in range(len(input_list)) for key, val in input_list[i].items()}
    return out_dict
