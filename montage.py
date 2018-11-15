import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib
import numpy as np

# [rows, cols]
def montage(img_list, layout=[4,4], img_size=4):
    
    fig = plt.figure()
    fig.set_size_inches(img_size*layout[1], img_size*layout[0])
    spec = gridspec.GridSpec(nrows=layout[0], ncols=layout[1], wspace=0, hspace=0)

    for i, img_data in enumerate(img_list):
        row = int(i / layout[1])
        col = int(i % layout[1])

        sub = fig.add_subplot(spec[row, col])
        sub.imshow(img_data, aspect='auto')
        sub.axis('off')

def montage_heatmap(img_list, layout=[4,4], img_size=4):
    
    fig = plt.figure()
    fig.set_size_inches(img_size*layout[1], img_size*layout[0])
    spec = gridspec.GridSpec(nrows=layout[0], ncols=layout[1], wspace=0, hspace=0)

    a = np.random.rand(256, 3)
    # make 0.0 black for background
    a[0,:] = np.zeros((1,3))
    cmap_rand = matplotlib.colors.ListedColormap(a)

    for i, img_data in enumerate(img_list):
        row = int(i / layout[1])
        col = int(i % layout[1])

        sub = fig.add_subplot(spec[row, col])
        #sub.imshow(img_data, aspect='auto', cmap=cmap_rand, interpolation='nearest')
        im = sub.imshow(img_data, aspect='auto', norm=matplotlib.colors.LogNorm(), cmap='plasma', interpolation='nearest')
        #im = sub.imshow(img_data, aspect='auto', cmap='plasma', interpolation='bilinear')
        sub.axis('off')
         
def montage_with_mask(img_list, mask_list, layout=[4,4], img_size=4):
    
    fig = plt.figure()
    fig.set_size_inches(img_size*layout[1], img_size*layout[0])
    spec = gridspec.GridSpec(nrows=layout[0], ncols=layout[1], wspace=0, hspace=0)

    for i, (img_data, mask_data) in enumerate(zip(img_list, mask_list)):
        row = int(i / layout[1])
        col = int(i % layout[1])

        if len(mask_data.shape) == 2:
            mask_data = np.expand_dims(mask_data, axis=-1)
        sub = fig.add_subplot(spec[row, col])
        sub.imshow(np.clip(img_data + 0.4*mask_data, a_min=0.0, a_max=1.0), aspect='auto', cmap='hot', interpolation='nearest')
        sub.axis('off')
