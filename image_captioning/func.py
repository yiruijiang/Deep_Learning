import zipfile
import json
from collections import defaultdict
import matplotlib.pyplot as plt
import utils

# .json is available from https://github.com/pveerina/imgcap/tree/master/data/mscoco

def get_captions_for_fns(fns, zip_fn, zip_json_path):
    zf = zipfile.ZipFile(zip_fn)
    j = json.loads(zf.read(zip_json_path).decode("utf8"))
    id_to_fn = {img["id"]: img["file_name"] for img in j["images"]} # extract info from json file and create a dict
    #test = {cap["image_id"]: cap["caption"] for cap in j["annotations"]}
    fn_to_caps = defaultdict(list) # we need the default factory method
    for cap in j['annotations']:
        fn_to_caps[id_to_fn[cap['image_id']]].append(cap['caption'])
    # create fn -> caption map
    fn_to_caps = dict(fn_to_caps)
    return list(map(lambda x: fn_to_caps[x], fns))

def show_training_example(train_img_fns, train_captions, example_idx = 0):
    zf = zipfile.ZipFile("train2014_sample.zip")
    captions_by_file = dict(zip(train_img_fns, train_captions))
    all_files = set(train_img_fns)
    found_files = list(filter(lambda x: x.filename.rsplit("/")[-1] in all_files, zf.filelist)) # the last word i.e. file name
    example = found_files[example_idx]
    print(example)
    img = utils.decode_image_from_buf(zf.read(example)) # example is ZipInfo necessary for zf.read() function
    plt.imshow(utils.image_center_crop(img))
    print(example.filename.rsplit("/")[-1])
    plt.title("\n".join(captions_by_file[example.filename.rsplit("/")[-1]]))
    plt.show()
