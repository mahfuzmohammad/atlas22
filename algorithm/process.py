import os
import numpy as np

import torch
from torch.cuda.amp import autocast

from monai import transforms
from monai.networks import one_hot
from monai.inferers import SlidingWindowInferer

import sys
sys.path.append('.')

import numpy as np
from settings import loader_settings
import medpy.io
import os, pathlib


def ensemble_model_list(alllogits, enmode='mean', enmax=5, softmax=False):

    if len(alllogits)==1:
        return alllogits[0]
    
    if enmode=='prob':
        vals_all = []
        num_classes = alllogits[0].shape[1] 
        
        num_probs = len(alllogits)
        for i in range(num_probs):
            allsoftmax = alllogits[i]
            if not softmax:         
                allsoftmax = torch.softmax(alllogits[i], dim=1)
            mask = torch.argmax(allsoftmax, dim=1, keepdim=True)
            mask = one_hot(mask, num_classes=num_classes, dim=1)
            vals = torch.mean(mask[:,1:] * allsoftmax[:,1:])
            vals_all.append(vals)
        
        mask = None
        allsoftmax = None

        vals_all = torch.stack(vals_all)
        vals2 = torch.argsort(vals_all, descending=True)[:enmax]
        vals2 = list(vals2.cpu().numpy())

        logits = 0
        for i in range(len(vals2)):
            logits += alllogits[vals2[i]]
        logits = logits/len(vals2)
        
   
    elif enmode=='mean':
        alllogits = torch.stack(alllogits, dim=0)
        logits = torch.mean(alllogits, dim=0)
    else:
        raise ValueError('unkown emmode'+str(enmode))

    ###

    if torch.sum(logits[:,1]>0.5)==0:
        alllogits2=[]
        for l in alllogits:
            if torch.sum(l[:,1]>0.5)>0:
                alllogits2.append(l)
        if len(alllogits2)>0:
            logits = sum(alllogits2)/len(alllogits2)

    return logits



class Seg():
    def __init__(self):
        # super().__init__(
        #     validators=dict(
        #         input_image=(
        #             UniqueImagesValidator(),
        #             UniquePathIndicesValidator(),
        #         )
        #     ),
        # )
        return
        
    @torch.no_grad()
    def process(self):

        np.set_printoptions(formatter={'float': '{: 0.3f}'.format}, suppress=True)
        torch.backends.cudnn.benchmark = True

        dirname = os.path.dirname(__file__)
        checkpoints = [ 
                        os.path.join(dirname, 'ts/model0.ts'),
                        os.path.join(dirname, 'ts/model1.ts'),
                        os.path.join(dirname, 'ts/model2.ts'),
                        os.path.join(dirname, 'ts/model3.ts'),
                        os.path.join(dirname, 'ts/model4.ts'),

                        os.path.join(dirname, 'ts/model5.ts'),
                        os.path.join(dirname, 'ts/model6.ts'),
                        os.path.join(dirname, 'ts/model7.ts'),
                        os.path.join(dirname, 'ts/model8.ts'),
                        os.path.join(dirname, 'ts/model9.ts'),

                        ]


        model_inferer = SlidingWindowInferer(roi_size=[192, 224, 144], overlap=0.625, mode='gaussian', cache_roi_weight_map=True, sw_batch_size=1)

        t = transforms.ScaleIntensityRange(a_min=4, a_max=100, b_min=0, b_max=1, clip=True)
        t2 = transforms.NormalizeIntensity(nonzero=True, channel_wise=True)


        inp_path = loader_settings['InputPath']  # Path for the input
        out_path = loader_settings['OutputPath']  # Path for the output
        file_list = os.listdir(inp_path)  # List of files in the input
        file_list = [os.path.join(inp_path, f) for f in file_list]
        for fil in file_list:
            dat, hdr = medpy.io.load(fil)  # dat is a numpy array
            im_shape = dat.shape
            dat = dat.reshape(1, 1, *im_shape)  # reshape to Pytorch standard

            image = torch.from_numpy(dat).float().cuda(0)
            image = t2(t(image)) #normalize

            all_probs=[]
            i=0
            for checkpoint in checkpoints:
                print('Inference with', checkpoint)

                model = torch.jit.load(checkpoint)
                model.cuda(0)
                model.eval()
 
                with autocast(enabled=True):
                    logits = model_inferer(inputs=image, network=model) 

                probs = torch.softmax(logits.float(), dim=1).cpu()
                all_probs.append(probs)
                i=i+1

            probs = ensemble_model_list(all_probs, enmode='prob', enmax=5, softmax=True)

            dat = torch.argmax(probs, dim=1).cpu().numpy().astype(np.uint8)
            dat = dat[0]

            ###########
            # dat = dat.reshape(*im_shape)
            out_name = os.path.basename(fil)
            out_filepath = os.path.join(out_path, out_name)
            print(f'=== saving {out_filepath} from {fil} ===')
            medpy.io.save(dat, out_filepath, hdr=hdr)

        return


if __name__ == "__main__":
    pathlib.Path("/output/images/stroke-lesion-segmentation/").mkdir(parents=True, exist_ok=True)
    Seg().process()

