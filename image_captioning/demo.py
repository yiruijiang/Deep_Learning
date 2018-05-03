import func
import utils

train_img_embeds = utils.read_pickle("train_img_embeds.pickle")
train_img_fns = utils.read_pickle("train_img_fns.pickle")
val_img_embeds = utils.read_pickle("val_img_embeds.pickle")
val_img_fns = utils.read_pickle("val_img_fns.pickle")

train_captions = func.get_captions_for_fns(train_img_fns, "captions_train-val2014.zip", 
                                       "annotations/captions_train2014.json")

func.show_training_example(train_img_fns, train_captions, example_idx = 99)
