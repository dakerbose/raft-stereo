'''
RAFTStereo                                         [1, 1, 640, 960]          --
├─MultiBasicEncoder: 1-1                           [1, 128, 160, 240]        --
│    └─Conv2d: 2-1                                 [1, 64, 640, 960]         9,472
│    └─BatchNorm2d: 2-2                            [1, 64, 640, 960]         128
│    └─ReLU: 2-3                                   [1, 64, 640, 960]         --
│    └─Sequential: 2-4                             [1, 64, 640, 960]         --
│    │    └─ResidualBlock: 3-1                     [1, 64, 640, 960]         74,112
│    │    └─ResidualBlock: 3-2                     [1, 64, 640, 960]         74,112
│    └─Sequential: 2-5                             [1, 96, 320, 480]         --
│    │    └─ResidualBlock: 3-3                     [1, 96, 320, 480]         145,248
│    │    └─ResidualBlock: 3-4                     [1, 96, 320, 480]         166,464
│    └─Sequential: 2-6                             [1, 128, 160, 240]        --
│    │    └─ResidualBlock: 3-5                     [1, 128, 160, 240]        271,488
│    │    └─ResidualBlock: 3-6                     [1, 128, 160, 240]        295,680
│    └─ModuleList: 2-7                             --                        --
│    │    └─Sequential: 3-7                        [1, 128, 160, 240]        443,264
│    │    └─Sequential: 3-8                        [1, 128, 160, 240]        443,264
│    └─Sequential: 2-8                             [1, 128, 80, 120]         --
│    │    └─ResidualBlock: 3-9                     [1, 128, 80, 120]         312,448
│    │    └─ResidualBlock: 3-10                    [1, 128, 80, 120]         295,680
│    └─ModuleList: 2-9                             --                        --
│    │    └─Sequential: 3-11                       [1, 128, 80, 120]         443,264
│    │    └─Sequential: 3-12                       [1, 128, 80, 120]         443,264
│    └─Sequential: 2-10                            [1, 128, 40, 60]          --
│    │    └─ResidualBlock: 3-13                    [1, 128, 40, 60]          312,448
│    │    └─ResidualBlock: 3-14                    [1, 128, 40, 60]          295,680
│    └─ModuleList: 2-11                            --                        --
│    │    └─Conv2d: 3-15                           [1, 128, 40, 60]          147,584
│    │    └─Conv2d: 3-16                           [1, 128, 40, 60]          147,584
'''
ModuleList用于不同分辨率输出，context_dims和hidden_dims都是[128, 128, 128]
也就是说Sequential: 2-6里面有两层ResidualBlock，而ModuleList里面也有也有一层ResidualBlock
'''
        output_list = []
        for dim in output_dim:
            conv_out = nn.Sequential(
                ResidualBlock(128, 128, self.norm_fn, stride=1),
                nn.Conv2d(128, dim[2], 3, padding=1))
            output_list.append(conv_out)

        self.outputs08 = nn.ModuleList(output_list)
'''
