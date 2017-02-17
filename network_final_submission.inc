# Definition of network architecture and other model parameters
# The network has a total of 373169 weights.

# number of models for bagging
model_count=5

# length of spectrogram slice (must correspond to architecture below)
net_width=700

nl=leaky_rectify
# augmentation
featpart="augment:Rotation|augment:Shift(low=-1,high=1,axis=3)"
# normalization
normpart="normalize:BatchNorm(axes=0x1x2,alpha=0.1,beta=None,gamma=None)"  # normalize over all except frequency axis
# convolutional layers
convpart="Conv2D(32x5x3)|${nl}|Pool2D(3x3)|Conv2D(32x5x3)|${nl}|Pool2D(3x3)|Conv2D(64x3x3)|${nl}|Pool2D(3x3)|Conv2D(128x3x1)|${nl}|Pool2D(3x2)|Conv2D(256x3x1)|${nl}|Pool2D(3x1)"
# dense layers
densepart="Dropout(p=0.5)|Dense(1)|output:sigmoid"
# all together
net_layers="${featpart}|${normpart}|${convpart}|${densepart}"

# training options
net_options="--problem=binary --var measures=auroc --var updates=adam --var batchsize=64 --var epochsize=1500 --var epochsize_val=500 --var learning_rate_mode=adaptive --var learning_rate=0.001 --var learning_rate_stop=0.00001"


# number of folds for pseudo-labeling
pseudo_folds=2
# confidence threshold for pseudo-labeling
pseudo_threshold=0.1
