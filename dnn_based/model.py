import torch.nn as nn


class UNet(nn.Module):
    def __init__(self, tcontext, fft_size, dropout, batchnorm):
        print('Using a U-NET')
        self.tcontext = tcontext
        self.fft_size = fft_size
        self.dropout=dropout
        self.batchnorm=batchnorm
        super(UNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, (5, 5), stride=(1, 2))
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, (5, 5), stride=(1, 2))
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, (5, 5), stride=2)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, (5, 5), stride=2)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 256, (5, 5), stride=2)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 512, (5, 5), stride=2)
        self.bn6 = nn.BatchNorm2d(512)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2)
        self.drop1 = nn.Dropout2d(p=0.5)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.drop3 = nn.Dropout2d(p=0.5)
        self.drop4 = nn.Dropout2d(p=0.5)
        self.drop5 = nn.Dropout2d(p=0.5)

        self.fc1 = nn.Linear(512 * 4 * 5, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 4)

    def forward(self, x):
        x = self.conv1(x)
        print(x.shape)
        if self.batchnorm:
            x = self.bn1(x)
        x = self.lrelu(x)
        if self.dropout:
            x = self.drop1(x)
        x = self.conv2(x)
        print(x.shape)
        if self.batchnorm:
            x = self.bn2(x)
        x = self.lrelu(x)
        if self.dropout:
            x = self.drop2(x)
        x = self.conv3(x)
        print(x.shape)
        if self.batchnorm:
            x = self.bn3(x)
        if self.dropout:
            x = self.drop3(x)
        x = self.lrelu(x)
        x = self.conv4(x)
        print(x.shape)
        if self.batchnorm:
            x = self.bn4(x)
        x = self.lrelu(x)
        x = self.conv5(x)
        print(x.shape)
        if self.batchnorm:
            x = self.bn5(x)
        x = self.lrelu(x)
        x = self.conv6(x)
        print(x.shape)
        if self.batchnorm:
            x = self.bn6(x)
        x = self.lrelu(x)
        x = x.view(-1, 512 * 4 * 5)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x.squeeze()

#TODO: rebuild resnet using classes of sub-blocks
class resnet152(nn.Module):
    def __init__(self, tcontext, fft_size, dropout, batchnorm):
        print('Using a ResNet')
        self.tcontext = tcontext
        self.fft_size = fft_size
        self.dropout=dropout
        self.batchnorm=batchnorm
        super(resnet152, self).__init__()

        #zero stage
        self.conv1 = nn.Conv2d(1, 64, (7, 7), stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(3, stride=2)

        #first stage
        self.conv2_1 = nn.Conv2d(64, 64, (1, 1), padding=1)
        self.bn2_1 = nn.BatchNorm2d(64)
        self.conv2_2 = nn.Conv2d(64, 64, (3, 3))
        self.bn2_2 = nn.BatchNorm2d(64)
        self.conv2_3 = nn.Conv2d(64, 256, (1,1))
        self.bn2_3 = nn.BatchNorm2d(256)
        self.aux2_1 = nn.Conv2d(64, 256, 1)

        self.conv2_4 = nn.Conv2d(256, 64, (1, 1), padding=1)
        self.bn2_4 = nn.BatchNorm2d(64)
        self.conv2_5 = nn.Conv2d(64, 64, (3, 3))
        self.bn2_5 = nn.BatchNorm2d(64)
        self.conv2_6 = nn.Conv2d(64, 256, (1,1))
        self.bn2_6 = nn.BatchNorm2d(256)

        self.conv2_7 = nn.Conv2d(256, 64, (1, 1), padding=1)
        self.bn2_7 = nn.BatchNorm2d(64)
        self.conv2_8 = nn.Conv2d(64, 64, (3, 3))
        self.bn2_8 = nn.BatchNorm2d(64)
        self.conv2_9 = nn.Conv2d(64, 256, (1,1))
        self.bn2_9 = nn.BatchNorm2d(256)
        self.aux2_3 = nn.Conv2d(64, 256, 1)

        #second stage
        self.conv3_1 = nn.Conv2d(256, 128, 1, padding=1)
        self.bn3_1 = nn.BatchNorm2d(128)
        self.conv3_2 = nn.Conv2d(128, 128, (3,3))
        self.bn3_2 = nn.BatchNorm2d(128)
        self.conv3_3 = nn.Conv2d(128, 512, 1)
        self.bn3_3 = nn.BatchNorm2d(512)
        self.aux3_1 = nn.Conv2d(256, 512, 1)

        self.conv3_4 = nn.Conv2d(512, 128, 1, padding=1)
        self.bn3_4 = nn.BatchNorm2d(128)
        self.conv3_5 = nn.Conv2d(128, 128, (3,3))
        self.bn3_5 = nn.BatchNorm2d(128)
        self.conv3_6 = nn.Conv2d(128, 512, 1)
        self.bn3_6 = nn.BatchNorm2d(512)

        self.conv3_7 = nn.Conv2d(512, 128, 1, padding=1)
        self.bn3_7 = nn.BatchNorm2d(128)
        self.conv3_8 = nn.Conv2d(128, 128, (3,3))
        self.bn3_8 = nn.BatchNorm2d(128)
        self.conv3_9 = nn.Conv2d(128, 512, (1,1))
        self.bn3_9 = nn.BatchNorm2d(512)

        self.conv3_10 = nn.Conv2d(512, 128, 1, padding=1)
        self.bn3_10 = nn.BatchNorm2d(128)
        self.conv3_11 = nn.Conv2d(128, 128, (3,3))
        self.bn3_11 = nn.BatchNorm2d(128)
        self.conv3_12 = nn.Conv2d(128, 512, (1,1))
        self.bn3_12 = nn.BatchNorm2d(512)

        self.conv3_13 = nn.Conv2d(512, 128, 1, padding=1)
        self.bn3_13 = nn.BatchNorm2d(128)
        self.conv3_14 = nn.Conv2d(128, 128, (3,3))
        self.bn3_14 = nn.BatchNorm2d(128)
        self.conv3_15 = nn.Conv2d(128, 512, (1,1))
        self.bn3_15 = nn.BatchNorm2d(512)

        self.conv3_16 = nn.Conv2d(512, 128, 1, padding=1)
        self.bn3_16 = nn.BatchNorm2d(128)
        self.conv3_17 = nn.Conv2d(128, 128, (3,3))
        self.bn3_17 = nn.BatchNorm2d(128)
        self.conv3_18 = nn.Conv2d(128, 512, (1,1))
        self.bn3_18 = nn.BatchNorm2d(512)

        self.conv3_19 = nn.Conv2d(512, 128, 1, padding=1)
        self.bn3_19 = nn.BatchNorm2d(128)
        self.conv3_20 = nn.Conv2d(128, 128, (3,3))
        self.bn3_20 = nn.BatchNorm2d(128)
        self.conv3_21 = nn.Conv2d(128, 512, (1,1))
        self.bn3_21 = nn.BatchNorm2d(512)

        self.conv3_22 = nn.Conv2d(512, 128, 1, padding=1)
        self.bn3_22 = nn.BatchNorm2d(128)
        self.conv3_23 = nn.Conv2d(128, 128, (3,3))
        self.bn3_23 = nn.BatchNorm2d(128)
        self.conv3_24 = nn.Conv2d(128, 512, (1,1))
        self.bn3_24 = nn.BatchNorm2d(512)

        #THIRD STAGE
        self.conv4_1 = nn.Conv2d(512, 256, 1, padding=1)
        self.bn4_1 = nn.BatchNorm2d(256)
        self.conv4_2 = nn.Conv2d(256, 256, (3, 3))
        self.bn4_2 = nn.BatchNorm2d(256)
        self.conv4_3 = nn.Conv2d(256, 1024, (1, 1))
        self.bn4_3 = nn.BatchNorm2d(1024)
        self.aux4 = nn.Conv2d(512, 1024, 1)

        self.conv4_4 = nn.Conv2d(1024, 256, 1, padding=1)
        self.bn4_4 = nn.BatchNorm2d(256)
        self.conv4_5 = nn.Conv2d(256, 256, (3, 3))
        self.bn4_5 = nn.BatchNorm2d(256)
        self.conv4_6 = nn.Conv2d(256, 1024, (1, 1))
        self.bn4_6 = nn.BatchNorm2d(1024)

        self.conv4_7 = nn.Conv2d(1024, 256, 1, padding=1)
        self.bn4_7 = nn.BatchNorm2d(256)
        self.conv4_8 = nn.Conv2d(256, 256, (3, 3))
        self.bn4_8 = nn.BatchNorm2d(256)
        self.conv4_9 = nn.Conv2d(256, 1024, (1, 1))
        self.bn4_9 = nn.BatchNorm2d(1024)

        self.conv4_10 = nn.Conv2d(1024, 256, 1, padding=1)
        self.bn4_10 = nn.BatchNorm2d(256)
        self.conv4_11 = nn.Conv2d(256, 256, (3, 3))
        self.bn4_11 = nn.BatchNorm2d(256)
        self.conv4_12 = nn.Conv2d(256, 1024, (1, 1))
        self.bn4_12 = nn.BatchNorm2d(1024)

        self.conv4_13 = nn.Conv2d(1024, 256, 1, padding=1)
        self.bn4_13 = nn.BatchNorm2d(256)
        self.conv4_14 = nn.Conv2d(256, 256, (3, 3))
        self.bn4_14 = nn.BatchNorm2d(256)
        self.conv4_15 = nn.Conv2d(256, 1024, (1, 1))
        self.bn4_15 = nn.BatchNorm2d(1024)

        self.conv4_16 = nn.Conv2d(1024, 256, 1, padding=1)
        self.bn4_16 = nn.BatchNorm2d(256)
        self.conv4_17 = nn.Conv2d(256, 256, (3, 3))
        self.bn4_17 = nn.BatchNorm2d(256)
        self.conv4_18 = nn.Conv2d(256, 1024, (1, 1))
        self.bn4_18 = nn.BatchNorm2d(1024)

        self.conv4_19 = nn.Conv2d(1024, 256, 1, padding=1)
        self.bn4_19 = nn.BatchNorm2d(256)
        self.conv4_20 = nn.Conv2d(256, 256, (3, 3))
        self.bn4_20 = nn.BatchNorm2d(256)
        self.conv4_21 = nn.Conv2d(256, 1024, (1, 1))
        self.bn4_21 = nn.BatchNorm2d(1024)

        self.conv4_22 = nn.Conv2d(1024, 256, 1, padding=1)
        self.bn4_22 = nn.BatchNorm2d(256)
        self.conv4_23 = nn.Conv2d(256, 256, (3, 3))
        self.bn4_23 = nn.BatchNorm2d(256)
        self.conv4_24 = nn.Conv2d(256, 1024, (1, 1))
        self.bn4_24 = nn.BatchNorm2d(1024)

        self.conv4_25 = nn.Conv2d(1024, 256, 1, padding=1)
        self.bn4_25 = nn.BatchNorm2d(256)
        self.conv4_26 = nn.Conv2d(256, 256, (3, 3))
        self.bn4_26 = nn.BatchNorm2d(256)
        self.conv4_27 = nn.Conv2d(256, 1024, (1, 1))
        self.bn4_27 = nn.BatchNorm2d(1024)

        self.conv4_28 = nn.Conv2d(1024, 256, 1, padding=1)
        self.bn4_28 = nn.BatchNorm2d(256)
        self.conv4_29 = nn.Conv2d(256, 256, (3, 3))
        self.bn4_29 = nn.BatchNorm2d(256)
        self.conv4_30 = nn.Conv2d(256, 1024, (1, 1))
        self.bn4_30 = nn.BatchNorm2d(1024)

        self.conv4_31 = nn.Conv2d(1024, 256, 1, padding=1)
        self.bn4_31 = nn.BatchNorm2d(256)
        self.conv4_32 = nn.Conv2d(256, 256, (3, 3))
        self.bn4_32 = nn.BatchNorm2d(256)
        self.conv4_33 = nn.Conv2d(256, 1024, (1, 1))
        self.bn4_33 = nn.BatchNorm2d(1024)

        self.conv4_34 = nn.Conv2d(1024, 256, 1, padding=1)
        self.bn4_34 = nn.BatchNorm2d(256)
        self.conv4_35 = nn.Conv2d(256, 256, (3, 3))
        self.bn4_35 = nn.BatchNorm2d(256)
        self.conv4_36 = nn.Conv2d(256, 1024, (1, 1))
        self.bn4_36 = nn.BatchNorm2d(1024)

        self.conv4_37 = nn.Conv2d(1024, 256, 1, padding=1)
        self.bn4_37 = nn.BatchNorm2d(256)
        self.conv4_38 = nn.Conv2d(256, 256, (3, 3))
        self.bn4_38 = nn.BatchNorm2d(256)
        self.conv4_39 = nn.Conv2d(256, 1024, (1, 1))
        self.bn4_39 = nn.BatchNorm2d(1024)

        self.conv4_40 = nn.Conv2d(1024, 256, 1, padding=1)
        self.bn4_40 = nn.BatchNorm2d(256)
        self.conv4_41 = nn.Conv2d(256, 256, (3, 3))
        self.bn4_41 = nn.BatchNorm2d(256)
        self.conv4_42 = nn.Conv2d(256, 1024, (1, 1))
        self.bn4_42 = nn.BatchNorm2d(1024)

        self.conv4_43 = nn.Conv2d(1024, 256, 1, padding=1)
        self.bn4_43 = nn.BatchNorm2d(256)
        self.conv4_44 = nn.Conv2d(256, 256, (3, 3))
        self.bn4_44 = nn.BatchNorm2d(256)
        self.conv4_45 = nn.Conv2d(256, 1024, (1, 1))
        self.bn4_45 = nn.BatchNorm2d(1024)

        self.conv4_46 = nn.Conv2d(1024, 256, 1, padding=1)
        self.bn4_46 = nn.BatchNorm2d(256)
        self.conv4_47 = nn.Conv2d(256, 256, (3, 3))
        self.bn4_47 = nn.BatchNorm2d(256)
        self.conv4_48 = nn.Conv2d(256, 1024, (1, 1))
        self.bn4_48 = nn.BatchNorm2d(1024)

        self.conv4_49 = nn.Conv2d(1024, 256, 1, padding=1)
        self.bn4_49 = nn.BatchNorm2d(256)
        self.conv4_50 = nn.Conv2d(256, 256, (3, 3))
        self.bn4_50 = nn.BatchNorm2d(256)
        self.conv4_51 = nn.Conv2d(256, 1024, (1, 1))
        self.bn4_51 = nn.BatchNorm2d(1024)

        self.conv4_52 = nn.Conv2d(1024, 256, 1, padding=1)
        self.bn4_52 = nn.BatchNorm2d(256)
        self.conv4_53 = nn.Conv2d(256, 256, (3, 3))
        self.bn4_53 = nn.BatchNorm2d(256)
        self.conv4_54 = nn.Conv2d(256, 1024, (1, 1))
        self.bn4_54 = nn.BatchNorm2d(1024)

        self.conv4_55 = nn.Conv2d(1024, 256, 1, padding=1)
        self.bn4_55 = nn.BatchNorm2d(256)
        self.conv4_56 = nn.Conv2d(256, 256, (3, 3))
        self.bn4_56 = nn.BatchNorm2d(256)
        self.conv4_57 = nn.Conv2d(256, 1024, (1, 1))
        self.bn4_57 = nn.BatchNorm2d(1024)

        self.conv4_58 = nn.Conv2d(1024, 256, 1, padding=1)
        self.bn4_58 = nn.BatchNorm2d(256)
        self.conv4_59 = nn.Conv2d(256, 256, (3, 3))
        self.bn4_59 = nn.BatchNorm2d(256)
        self.conv4_60 = nn.Conv2d(256, 1024, (1, 1))
        self.bn4_60 = nn.BatchNorm2d(1024)

        self.conv4_61 = nn.Conv2d(1024, 256, 1, padding=1)
        self.bn4_61 = nn.BatchNorm2d(256)
        self.conv4_62 = nn.Conv2d(256, 256, (3, 3))
        self.bn4_62 = nn.BatchNorm2d(256)
        self.conv4_63 = nn.Conv2d(256, 1024, (1, 1))
        self.bn4_63 = nn.BatchNorm2d(1024)

        self.conv4_64 = nn.Conv2d(1024, 256, 1, padding=1)
        self.bn4_64 = nn.BatchNorm2d(256)
        self.conv4_65 = nn.Conv2d(256, 256, (3, 3))
        self.bn4_65 = nn.BatchNorm2d(256)
        self.conv4_66 = nn.Conv2d(256, 1024, (1, 1))
        self.bn4_66 = nn.BatchNorm2d(1024)

        self.conv4_67 = nn.Conv2d(1024, 256, 1, padding=1)
        self.bn4_67 = nn.BatchNorm2d(256)
        self.conv4_68 = nn.Conv2d(256, 256, (3, 3))
        self.bn4_68 = nn.BatchNorm2d(256)
        self.conv4_69 = nn.Conv2d(256, 1024, (1, 1))
        self.bn4_69 = nn.BatchNorm2d(1024)

        self.conv4_70 = nn.Conv2d(1024, 256, 1, padding=1)
        self.bn4_70 = nn.BatchNorm2d(256)
        self.conv4_71 = nn.Conv2d(256, 256, (3, 3))
        self.bn4_71 = nn.BatchNorm2d(256)
        self.conv4_72 = nn.Conv2d(256, 1024, (1, 1))
        self.bn4_72 = nn.BatchNorm2d(1024)

        self.conv4_73 = nn.Conv2d(1024, 256, 1, padding=1)
        self.bn4_73 = nn.BatchNorm2d(256)
        self.conv4_74 = nn.Conv2d(256, 256, (3, 3))
        self.bn4_74 = nn.BatchNorm2d(256)
        self.conv4_75 = nn.Conv2d(256, 1024, (1, 1))
        self.bn4_75 = nn.BatchNorm2d(1024)

        self.conv4_76 = nn.Conv2d(1024, 256, 1, padding=1)
        self.bn4_76 = nn.BatchNorm2d(256)
        self.conv4_77 = nn.Conv2d(256, 256, (3, 3))
        self.bn4_77 = nn.BatchNorm2d(256)
        self.conv4_78 = nn.Conv2d(256, 1024, (1, 1))
        self.bn4_78 = nn.BatchNorm2d(1024)

        self.conv4_79 = nn.Conv2d(1024, 256, 1, padding=1)
        self.bn4_79 = nn.BatchNorm2d(256)
        self.conv4_80 = nn.Conv2d(256, 256, (3, 3))
        self.bn4_80 = nn.BatchNorm2d(256)
        self.conv4_81 = nn.Conv2d(256, 1024, (1, 1))
        self.bn4_81 = nn.BatchNorm2d(1024)

        self.conv4_82 = nn.Conv2d(1024, 256, 1, padding=1)
        self.bn4_82 = nn.BatchNorm2d(256)
        self.conv4_83 = nn.Conv2d(256, 256, (3, 3))
        self.bn4_83 = nn.BatchNorm2d(256)
        self.conv4_84 = nn.Conv2d(256, 1024, (1, 1))
        self.bn4_84 = nn.BatchNorm2d(1024)

        self.conv4_85 = nn.Conv2d(1024, 256, 1, padding=1)
        self.bn4_85 = nn.BatchNorm2d(256)
        self.conv4_86 = nn.Conv2d(256, 256, (3, 3))
        self.bn4_86 = nn.BatchNorm2d(256)
        self.conv4_87 = nn.Conv2d(256, 1024, (1, 1))
        self.bn4_87 = nn.BatchNorm2d(1024)

        self.conv4_88 = nn.Conv2d(1024, 256, 1, padding=1)
        self.bn4_88 = nn.BatchNorm2d(256)
        self.conv4_89 = nn.Conv2d(256, 256, (3, 3))
        self.bn4_89 = nn.BatchNorm2d(256)
        self.conv4_90 = nn.Conv2d(256, 1024, (1, 1))
        self.bn4_90 = nn.BatchNorm2d(1024)

        self.conv4_91 = nn.Conv2d(1024, 256, 1, padding=1)
        self.bn4_91 = nn.BatchNorm2d(256)
        self.conv4_92 = nn.Conv2d(256, 256, (3, 3))
        self.bn4_92 = nn.BatchNorm2d(256)
        self.conv4_93 = nn.Conv2d(256, 1024, (1, 1))
        self.bn4_93 = nn.BatchNorm2d(1024)

        self.conv4_94 = nn.Conv2d(1024, 256, 1, padding=1)
        self.bn4_94 = nn.BatchNorm2d(256)
        self.conv4_95 = nn.Conv2d(256, 256, (3, 3))
        self.bn4_95 = nn.BatchNorm2d(256)
        self.conv4_96 = nn.Conv2d(256, 1024, (1, 1))
        self.bn4_96 = nn.BatchNorm2d(1024)

        self.conv4_97 = nn.Conv2d(1024, 256, 1, padding=1)
        self.bn4_97 = nn.BatchNorm2d(256)
        self.conv4_98 = nn.Conv2d(256, 256, (3, 3))
        self.bn4_98 = nn.BatchNorm2d(256)
        self.conv4_99 = nn.Conv2d(256, 1024, (1, 1))
        self.bn4_99 = nn.BatchNorm2d(1024)

        self.conv4_100 = nn.Conv2d(1024, 256, 1, padding=1)
        self.bn4_100 = nn.BatchNorm2d(256)
        self.conv4_101 = nn.Conv2d(256, 256, (3, 3))
        self.bn4_101 = nn.BatchNorm2d(256)
        self.conv4_102 = nn.Conv2d(256, 1024, (1, 1))
        self.bn4_102 = nn.BatchNorm2d(1024)

        self.conv4_103 = nn.Conv2d(1024, 256, 1, padding=1)
        self.bn4_103 = nn.BatchNorm2d(256)
        self.conv4_104 = nn.Conv2d(256, 256, (3, 3))
        self.bn4_104 = nn.BatchNorm2d(256)
        self.conv4_105 = nn.Conv2d(256, 1024, (1, 1))
        self.bn4_105 = nn.BatchNorm2d(1024)

        self.conv4_106 = nn.Conv2d(1024, 256, 1, padding=1)
        self.bn4_106 = nn.BatchNorm2d(256)
        self.conv4_107 = nn.Conv2d(256, 256, (3, 3))
        self.bn4_107 = nn.BatchNorm2d(256)
        self.conv4_108 = nn.Conv2d(256, 1024, (1, 1))
        self.bn4_108 = nn.BatchNorm2d(1024)

        # FOURTH STAGE
        self.conv5_1 = nn.Conv2d(1024, 512, 1, padding=1)
        self.bn5_1 = nn.BatchNorm2d(512)
        self.conv5_2 = nn.Conv2d(512, 512, (3, 3))
        self.bn5_2 = nn.BatchNorm2d(512)
        self.conv5_3 = nn.Conv2d(512, 2048, (1, 1))
        self.bn5_3 = nn.BatchNorm2d(2048)
        self.aux5 = nn.Conv2d(1024, 2048, 1)

        self.conv5_4 = nn.Conv2d(2048, 512, 1, padding=1)
        self.bn5_4 = nn.BatchNorm2d(512)
        self.conv5_5 = nn.Conv2d(512, 512, (3, 3))
        self.bn5_5 = nn.BatchNorm2d(512)
        self.conv5_6 = nn.Conv2d(512, 2048, (1, 1))
        self.bn5_6 = nn.BatchNorm2d(2048)

        self.conv5_7 = nn.Conv2d(2048, 512, 1, padding=1)
        self.bn5_7 = nn.BatchNorm2d(512)
        self.conv5_8 = nn.Conv2d(512, 512, (3, 3))
        self.bn5_8 = nn.BatchNorm2d(512)
        self.conv5_9 = nn.Conv2d(512, 2048, (1, 1))
        self.bn5_9 = nn.BatchNorm2d(2048)

        self.avgpool = nn.AvgPool2d((28,126))

        self.final = nn.Linear(2048, 4)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.pool1(nn.functional.relu(x))

        #FIRST STAGE
        x2 = self.conv2_1(x)
        x2 = nn.functional.relu(self.bn2_1(x2))
        x2 = self.conv2_2(x2)
        x2 = nn.functional.relu(self.bn2_2(x2))
        x2 = self.conv2_3(x2)
        x2 = nn.functional.relu(self.bn2_3(x2))
        x = self.aux2_1(x)
        x = nn.functional.relu(x + x2)

        x2 = self.conv2_4(x)
        x2 = nn.functional.relu(self.bn2_4(x2))
        x2 = self.conv2_5(x2)
        x2 = nn.functional.relu(self.bn2_5(x2))
        x2 = self.conv2_6(x2)
        x2 = nn.functional.relu(self.bn2_6(x2))
        x = nn.functional.relu(x + x2)

        x2 = self.conv2_7(x)
        x2 = nn.functional.relu(self.bn2_7(x2))
        x2 = self.conv2_8(x2)
        x2 = nn.functional.relu(self.bn2_8(x2))
        x2 = self.conv2_9(x2)
        x2 = nn.functional.relu(self.bn2_9(x2))
        x = nn.functional.relu(x + x2)

        #SECOND STAGE
        x2 = self.conv3_1(x)
        x2 = nn.functional.relu(self.bn3_1(x2))
        x2 = self.conv3_2(x2)
        x2 = nn.functional.relu(self.bn3_2(x2))
        x2 = self.conv3_3(x2)
        x2 = nn.functional.relu(self.bn3_3(x2))
        x = self.aux3_1(x)
        x = nn.functional.relu(x + x2)


        x2 = self.conv3_4(x)
        x2 = nn.functional.relu(self.bn3_4(x2))
        x2 = self.conv3_5(x2)
        x2 = nn.functional.relu(self.bn3_5(x2))
        x2 = self.conv3_6(x2)
        x2 = nn.functional.relu(self.bn3_6(x2))
        x = nn.functional.relu(x + x2)

        x2 = self.conv3_7(x)
        x2 = nn.functional.relu(self.bn3_7(x2))
        x2 = self.conv3_8(x2)
        x2 = nn.functional.relu(self.bn3_8(x2))
        x2 = self.conv3_9(x2)
        x2 = nn.functional.relu(self.bn3_9(x2))
        x = nn.functional.relu(x + x2)

        x2 = self.conv3_10(x)
        x2 = nn.functional.relu(self.bn3_10(x2))
        x2 = self.conv3_11(x2)
        x2 = nn.functional.relu(self.bn3_11(x2))
        x2 = self.conv3_12(x2)
        x2 = nn.functional.relu(self.bn3_12(x2))
        x = nn.functional.relu(x + x2)

        x2 = self.conv3_13(x)
        x2 = nn.functional.relu(self.bn3_13(x2))
        x2 = self.conv3_14(x2)
        x2 = nn.functional.relu(self.bn3_14(x2))
        x2 = self.conv3_15(x2)
        x2 = nn.functional.relu(self.bn3_15(x2))
        x = nn.functional.relu(x + x2)

        x2 = self.conv3_16(x)
        x2 = nn.functional.relu(self.bn3_16(x2))
        x2 = self.conv3_17(x2)
        x2 = nn.functional.relu(self.bn3_17(x2))
        x2 = self.conv3_18(x2)
        x2 = nn.functional.relu(self.bn3_18(x2))
        x = nn.functional.relu(x + x2)

        x2 = self.conv3_19(x)
        x2 = nn.functional.relu(self.bn3_19(x2))
        x2 = self.conv3_20(x2)
        x2 = nn.functional.relu(self.bn3_20(x2))
        x2 = self.conv3_21(x2)
        x2 = nn.functional.relu(self.bn3_21(x2))
        x = nn.functional.relu(x + x2)

        x2 = self.conv3_22(x)
        x2 = nn.functional.relu(self.bn3_22(x2))
        x2 = self.conv3_23(x2)
        x2 = nn.functional.relu(self.bn3_23(x2))
        x2 = self.conv3_24(x2)
        x2 = nn.functional.relu(self.bn3_24(x2))
        x = nn.functional.relu(x + x2)


        #THIRD STAGE

        x2 = self.conv4_1(x)
        x2 = nn.functional.relu(self.bn4_1(x2))
        x2 = self.conv4_2(x2)
        x2 = nn.functional.relu(self.bn4_2(x2))
        x2 = self.conv4_3(x2)
        x2 = nn.functional.relu(self.bn4_3(x2))
        x = self.aux4(x)
        x = nn.functional.relu(x + x2)

        x2 = self.conv4_4(x)
        x2 = nn.functional.relu(self.bn4_4(x2))
        x2 = self.conv4_5(x2)
        x2 = nn.functional.relu(self.bn4_5(x2))
        x2 = self.conv4_6(x2)
        x2 = nn.functional.relu(self.bn4_6(x2))
        x = nn.functional.relu(x + x2)

        x2 = self.conv4_7(x)
        x2 = nn.functional.relu(self.bn4_7(x2))
        x2 = self.conv4_8(x2)
        x2 = nn.functional.relu(self.bn4_8(x2))
        x2 = self.conv4_9(x2)
        x2 = nn.functional.relu(self.bn4_9(x2))
        x = nn.functional.relu(x + x2)

        x2 = self.conv4_10(x)
        x2 = nn.functional.relu(self.bn4_10(x2))
        x2 = self.conv4_11(x2)
        x2 = nn.functional.relu(self.bn4_11(x2))
        x2 = self.conv4_12(x2)
        x2 = nn.functional.relu(self.bn4_12(x2))
        x = nn.functional.relu(x + x2)

        x2 = self.conv4_13(x)
        x2 = nn.functional.relu(self.bn4_13(x2))
        x2 = self.conv4_14(x2)
        x2 = nn.functional.relu(self.bn4_14(x2))
        x2 = self.conv4_15(x2)
        x2 = nn.functional.relu(self.bn4_15(x2))
        x = nn.functional.relu(x + x2)

        x2 = self.conv4_16(x)
        x2 = nn.functional.relu(self.bn4_16(x2))
        x2 = self.conv4_17(x2)
        x2 = nn.functional.relu(self.bn4_17(x2))
        x2 = self.conv4_18(x2)
        x2 = nn.functional.relu(self.bn4_18(x2))
        x = nn.functional.relu(x + x2)

        x2 = self.conv4_19(x)
        x2 = nn.functional.relu(self.bn4_19(x2))
        x2 = self.conv4_20(x2)
        x2 = nn.functional.relu(self.bn4_20(x2))
        x2 = self.conv4_21(x2)
        x2 = nn.functional.relu(self.bn4_21(x2))
        x = nn.functional.relu(x + x2)

        x2 = self.conv4_22(x)
        x2 = nn.functional.relu(self.bn4_22(x2))
        x2 = self.conv4_23(x2)
        x2 = nn.functional.relu(self.bn4_23(x2))
        x2 = self.conv4_24(x2)
        x2 = nn.functional.relu(self.bn4_24(x2))
        x = nn.functional.relu(x + x2)

        x2 = self.conv4_25(x)
        x2 = nn.functional.relu(self.bn4_25(x2))
        x2 = self.conv4_26(x2)
        x2 = nn.functional.relu(self.bn4_26(x2))
        x2 = self.conv4_27(x2)
        x2 = nn.functional.relu(self.bn4_27(x2))
        x = nn.functional.relu(x + x2)

        x2 = self.conv4_28(x)
        x2 = nn.functional.relu(self.bn4_28(x2))
        x2 = self.conv4_29(x2)
        x2 = nn.functional.relu(self.bn4_29(x2))
        x2 = self.conv4_30(x2)
        x2 = nn.functional.relu(self.bn4_30(x2))
        x = nn.functional.relu(x + x2)

        x2 = self.conv4_31(x)
        x2 = nn.functional.relu(self.bn4_31(x2))
        x2 = self.conv4_32(x2)
        x2 = nn.functional.relu(self.bn4_32(x2))
        x2 = self.conv4_33(x2)
        x2 = nn.functional.relu(self.bn4_33(x2))
        x = nn.functional.relu(x + x2)

        x2 = self.conv4_34(x)
        x2 = nn.functional.relu(self.bn4_34(x2))
        x2 = self.conv4_35(x2)
        x2 = nn.functional.relu(self.bn4_35(x2))
        x2 = self.conv4_36(x2)
        x2 = nn.functional.relu(self.bn4_36(x2))
        x = nn.functional.relu(x + x2)

        x2 = self.conv4_37(x)
        x2 = nn.functional.relu(self.bn4_37(x2))
        x2 = self.conv4_38(x2)
        x2 = nn.functional.relu(self.bn4_38(x2))
        x2 = self.conv4_39(x2)
        x2 = nn.functional.relu(self.bn4_39(x2))
        x = nn.functional.relu(x + x2)

        x2 = self.conv4_40(x)
        x2 = nn.functional.relu(self.bn4_40(x2))
        x2 = self.conv4_41(x2)
        x2 = nn.functional.relu(self.bn4_41(x2))
        x2 = self.conv4_42(x2)
        x2 = nn.functional.relu(self.bn4_42(x2))
        x = nn.functional.relu(x + x2)

        x2 = self.conv4_43(x)
        x2 = nn.functional.relu(self.bn4_43(x2))
        x2 = self.conv4_44(x2)
        x2 = nn.functional.relu(self.bn4_44(x2))
        x2 = self.conv4_45(x2)
        x2 = nn.functional.relu(self.bn4_45(x2))
        x = nn.functional.relu(x + x2)

        x2 = self.conv4_46(x)
        x2 = nn.functional.relu(self.bn4_46(x2))
        x2 = self.conv4_47(x2)
        x2 = nn.functional.relu(self.bn4_47(x2))
        x2 = self.conv4_48(x2)
        x2 = nn.functional.relu(self.bn4_48(x2))
        x = nn.functional.relu(x + x2)

        x2 = self.conv4_49(x)
        x2 = nn.functional.relu(self.bn4_49(x2))
        x2 = self.conv4_50(x2)
        x2 = nn.functional.relu(self.bn4_50(x2))
        x2 = self.conv4_51(x2)
        x2 = nn.functional.relu(self.bn4_51(x2))
        x = nn.functional.relu(x + x2)

        x2 = self.conv4_52(x)
        x2 = nn.functional.relu(self.bn4_52(x2))
        x2 = self.conv4_53(x2)
        x2 = nn.functional.relu(self.bn4_53(x2))
        x2 = self.conv4_54(x2)
        x2 = nn.functional.relu(self.bn4_54(x2))
        x = nn.functional.relu(x + x2)

        x2 = self.conv4_55(x)
        x2 = nn.functional.relu(self.bn4_55(x2))
        x2 = self.conv4_56(x2)
        x2 = nn.functional.relu(self.bn4_56(x2))
        x2 = self.conv4_57(x2)
        x2 = nn.functional.relu(self.bn4_57(x2))
        x = nn.functional.relu(x + x2)

        x2 = self.conv4_58(x)
        x2 = nn.functional.relu(self.bn4_58(x2))
        x2 = self.conv4_59(x2)
        x2 = nn.functional.relu(self.bn4_59(x2))
        x2 = self.conv4_60(x2)
        x2 = nn.functional.relu(self.bn4_60(x2))
        x = nn.functional.relu(x + x2)

        x2 = self.conv4_61(x)
        x2 = nn.functional.relu(self.bn4_61(x2))
        x2 = self.conv4_62(x2)
        x2 = nn.functional.relu(self.bn4_62(x2))
        x2 = self.conv4_63(x2)
        x2 = nn.functional.relu(self.bn4_63(x2))
        x = nn.functional.relu(x + x2)

        x2 = self.conv4_64(x)
        x2 = nn.functional.relu(self.bn4_64(x2))
        x2 = self.conv4_65(x2)
        x2 = nn.functional.relu(self.bn4_65(x2))
        x2 = self.conv4_66(x2)
        x2 = nn.functional.relu(self.bn4_66(x2))
        x = nn.functional.relu(x + x2)

        x2 = self.conv4_67(x)
        x2 = nn.functional.relu(self.bn4_67(x2))
        x2 = self.conv4_68(x2)
        x2 = nn.functional.relu(self.bn4_68(x2))
        x2 = self.conv4_69(x2)
        x2 = nn.functional.relu(self.bn4_69(x2))
        x = nn.functional.relu(x + x2)

        x2 = self.conv4_70(x)
        x2 = nn.functional.relu(self.bn4_70(x2))
        x2 = self.conv4_71(x2)
        x2 = nn.functional.relu(self.bn4_71(x2))
        x2 = self.conv4_72(x2)
        x2 = nn.functional.relu(self.bn4_72(x2))
        x = nn.functional.relu(x + x2)

        x2 = self.conv4_73(x)
        x2 = nn.functional.relu(self.bn4_73(x2))
        x2 = self.conv4_74(x2)
        x2 = nn.functional.relu(self.bn4_74(x2))
        x2 = self.conv4_75(x2)
        x2 = nn.functional.relu(self.bn4_75(x2))
        x = nn.functional.relu(x + x2)

        x2 = self.conv4_76(x)
        x2 = nn.functional.relu(self.bn4_76(x2))
        x2 = self.conv4_77(x2)
        x2 = nn.functional.relu(self.bn4_77(x2))
        x2 = self.conv4_78(x2)
        x2 = nn.functional.relu(self.bn4_78(x2))
        x = nn.functional.relu(x + x2)

        x2 = self.conv4_79(x)
        x2 = nn.functional.relu(self.bn4_79(x2))
        x2 = self.conv4_80(x2)
        x2 = nn.functional.relu(self.bn4_80(x2))
        x2 = self.conv4_81(x2)
        x2 = nn.functional.relu(self.bn4_81(x2))
        x = nn.functional.relu(x + x2)

        x2 = self.conv4_82(x)
        x2 = nn.functional.relu(self.bn4_82(x2))
        x2 = self.conv4_83(x2)
        x2 = nn.functional.relu(self.bn4_83(x2))
        x2 = self.conv4_84(x2)
        x2 = nn.functional.relu(self.bn4_84(x2))
        x = nn.functional.relu(x + x2)

        x2 = self.conv4_85(x)
        x2 = nn.functional.relu(self.bn4_85(x2))
        x2 = self.conv4_86(x2)
        x2 = nn.functional.relu(self.bn4_86(x2))
        x2 = self.conv4_87(x2)
        x2 = nn.functional.relu(self.bn4_87(x2))
        x = nn.functional.relu(x + x2)

        x2 = self.conv4_88(x)
        x2 = nn.functional.relu(self.bn4_88(x2))
        x2 = self.conv4_89(x2)
        x2 = nn.functional.relu(self.bn4_89(x2))
        x2 = self.conv4_90(x2)
        x2 = nn.functional.relu(self.bn4_90(x2))
        x = nn.functional.relu(x + x2)

        x2 = self.conv4_91(x)
        x2 = nn.functional.relu(self.bn4_91(x2))
        x2 = self.conv4_92(x2)
        x2 = nn.functional.relu(self.bn4_92(x2))
        x2 = self.conv4_93(x2)
        x2 = nn.functional.relu(self.bn4_93(x2))
        x = nn.functional.relu(x + x2)

        x2 = self.conv4_94(x)
        x2 = nn.functional.relu(self.bn4_94(x2))
        x2 = self.conv4_95(x2)
        x2 = nn.functional.relu(self.bn4_95(x2))
        x2 = self.conv4_96(x2)
        x2 = nn.functional.relu(self.bn4_96(x2))
        x = nn.functional.relu(x + x2)

        x2 = self.conv4_97(x)
        x2 = nn.functional.relu(self.bn4_97(x2))
        x2 = self.conv4_98(x2)
        x2 = nn.functional.relu(self.bn4_98(x2))
        x2 = self.conv4_99(x2)
        x2 = nn.functional.relu(self.bn4_99(x2))
        x = nn.functional.relu(x + x2)

        x2 = self.conv4_100(x)
        x2 = nn.functional.relu(self.bn4_100(x2))
        x2 = self.conv4_101(x2)
        x2 = nn.functional.relu(self.bn4_101(x2))
        x2 = self.conv4_102(x2)
        x2 = nn.functional.relu(self.bn4_102(x2))
        x = nn.functional.relu(x + x2)

        x2 = self.conv4_103(x)
        x2 = nn.functional.relu(self.bn4_103(x2))
        x2 = self.conv4_104(x2)
        x2 = nn.functional.relu(self.bn4_104(x2))
        x2 = self.conv4_105(x2)
        x2 = nn.functional.relu(self.bn4_105(x2))
        x = nn.functional.relu(x + x2)

        x2 = self.conv4_106(x)
        x2 = nn.functional.relu(self.bn4_106(x2))
        x2 = self.conv4_107(x2)
        x2 = nn.functional.relu(self.bn4_107(x2))
        x2 = self.conv4_108(x2)
        x2 = nn.functional.relu(self.bn4_108(x2))
        x = nn.functional.relu(x + x2)


        # FOURTH STAGE

        x2 = self.conv5_1(x)
        x2 = nn.functional.relu(self.bn5_1(x2))
        x2 = self.conv5_2(x2)
        x2 = nn.functional.relu(self.bn5_2(x2))
        x2 = self.conv5_3(x2)
        x2 = nn.functional.relu(self.bn5_3(x2))
        x = self.aux5(x)
        x = nn.functional.relu(x + x2)

        x2 = self.conv5_4(x)
        x2 = nn.functional.relu(self.bn5_4(x2))
        x2 = self.conv5_5(x2)
        x2 = nn.functional.relu(self.bn5_5(x2))
        x2 = self.conv5_6(x2)
        x2 = nn.functional.relu(self.bn5_6(x2))
        x = nn.functional.relu(x + x2)

        x2 = self.conv5_7(x)
        x2 = nn.functional.relu(self.bn5_7(x2))
        x2 = self.conv5_8(x2)
        x2 = nn.functional.relu(self.bn5_8(x2))
        x2 = self.conv5_9(x2)
        x2 = nn.functional.relu(self.bn5_9(x2))
        x = nn.functional.relu(x + x2)

        x = self.avgpool(x)
        x = self.final(x.view(-1, 2048))

        return x.squeeze()

class resnet18(nn.Module):
    def __init__(self, tcontext, fft_size, dropout, batchnorm):
        print('Using a ResNet')
        self.tcontext = tcontext
        self.fft_size = fft_size
        self.dropout = dropout
        self.batchnorm = batchnorm
        super(resnet18, self).__init__()

        self.conv1 = nn.Conv2d(1, 64, (7, 7), stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(3, stride=2)

        self.conv2_1 = nn.Conv2d(64, 64, (3, 3), padding=(1))
        self.bn2_1 = nn.BatchNorm2d(64)
        self.conv2_2 = nn.Conv2d(64, 64, (3, 3), padding=1)
        self.bn2_2 = nn.BatchNorm2d(64)

        self.conv2_3 = nn.Conv2d(64, 64, (3, 3), padding=(1))
        self.bn2_3 = nn.BatchNorm2d(64)
        self.conv2_4 = nn.Conv2d(64, 64, (3, 3), padding=1)
        self.bn2_4 = nn.BatchNorm2d(64)



        # second stage
        self.conv3_1 = nn.Conv2d(64, 128, (3, 3), stride=2, padding=3)
        self.bn3_1 = nn.BatchNorm2d(128)
        self.conv3_2 = nn.Conv2d(128, 128, (3, 3))
        self.bn3_2 = nn.BatchNorm2d(128)

        self.conv3_3 = nn.Conv2d(128, 128, (3, 3), padding=1)
        self.bn3_3 = nn.BatchNorm2d(128)
        self.conv3_4 = nn.Conv2d(128, 128, (3, 3), padding=1)
        self.bn3_4 = nn.BatchNorm2d(128)



        self.auxc_1 = nn.Conv2d(64, 128, kernel_size=1, stride=2)

        # third stage
        self.conv4_1 = nn.Conv2d(128, 256, (3, 3), stride=2, padding=3)
        self.bn4_1 = nn.BatchNorm2d(256)
        self.conv4_2 = nn.Conv2d(256, 256, (3, 3))
        self.bn4_2 = nn.BatchNorm2d(256)

        self.conv4_3 = nn.Conv2d(256, 256, (3, 3), padding=1)
        self.bn4_3 = nn.BatchNorm2d(256)
        self.conv4_4 = nn.Conv2d(256, 256, (3, 3), padding=1)
        self.bn4_4 = nn.BatchNorm2d(256)


        self.auxc_2 = nn.Conv2d(128, 256, kernel_size=1, stride=2)

        # fourth stage
        self.conv5_1 = nn.Conv2d(256, 512, (3, 3), padding=1, stride=2)
        self.bn5_1 = nn.BatchNorm2d(512)
        self.conv5_2 = nn.Conv2d(512, 512, (3, 3), padding=1)
        self.bn5_2 = nn.BatchNorm2d(512)

        self.conv5_3 = nn.Conv2d(512, 512, (3, 3), padding=1)
        self.bn5_3 = nn.BatchNorm2d(512)
        self.conv5_4 = nn.Conv2d(512, 512, (3, 3), padding=1)
        self.bn5_4 = nn.BatchNorm2d(512)


        self.auxc5 = nn.Conv2d(256, 512, kernel_size=1, stride=2)

        self.avgpool = nn.AvgPool2d((4, 16))

        self.final = nn.Linear(512, 4)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.pool1(nn.functional.relu(x))

        #starts firt blocks
        x2 = self.conv2_1(x)
        x2 = nn.functional.relu(self.bn2_1(x2))
        x2 = self.conv2_2(x2)
        x2 = nn.functional.relu(self.bn2_2(x2))
        x = nn.functional.relu(x + x2)

        x2 = self.conv2_3(x)
        x2 = nn.functional.relu(self.bn2_3(x2))
        x2 = self.conv2_4(x2)
        x2 = nn.functional.relu(self.bn2_4(x2))
        x = nn.functional.relu(x + x2)




        #second stage
        x2 = self.conv3_1(x)
        x2 = nn.functional.relu(self.bn3_1(x2))
        x2 = self.conv3_2(x2)
        x2 = nn.functional.relu(self.bn3_2(x2))
        x = nn.functional.relu(self.auxc_1(x) + x2)

        x2 = self.conv3_3(x)
        x2 = nn.functional.relu(self.bn3_3(x2))
        x2 = self.conv3_4(x2)
        x2 = nn.functional.relu(self.bn3_4(x2))
        x = nn.functional.relu(x + x2)



        #thjrd stage
        x2 = self.conv4_1(x)
        x2 = nn.functional.relu(self.bn4_1(x2))
        x2 = self.conv4_2(x2)
        x2 = nn.functional.relu(self.bn4_2(x2))
        x = nn.functional.relu(self.auxc_2(x) + x2)

        x2 = self.conv4_3(x)
        x2 = nn.functional.relu(self.bn4_3(x2))
        x2 = self.conv4_4(x2)
        x2 = nn.functional.relu(self.bn4_4(x2))
        x = nn.functional.relu(x + x2)



        #fourth stage

        x2 = self.conv5_1(x)
        x2 = nn.functional.relu(self.bn5_1(x2))
        x2 = self.conv5_2(x2)
        x2 = nn.functional.relu(self.bn5_2(x2))
        x = self.auxc5(x)
        x = nn.functional.relu(x + x2)
        x2 = self.conv5_3(x)
        x2 = nn.functional.relu(self.bn5_3(x2))
        x2 = self.conv5_4(x2)
        x2 = nn.functional.relu(self.bn5_4(x2))
        x = nn.functional.relu(x + x2)


        x = self.avgpool(x)

        x = self.final(x.view(-1, 512))
        return x.squeeze()

class resnet34(nn.Module):
    def __init__(self, tcontext, fft_size, dropout, batchnorm):
        print('Using a ResNet')
        self.tcontext = tcontext
        self.fft_size = fft_size
        self.dropout = dropout
        self.batchnorm = batchnorm
        super(resnet34, self).__init__()

        self.conv1 = nn.Conv2d(1, 64, (7, 7), stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(3, stride=2)

        self.conv2_1 = nn.Conv2d(64, 64, (3, 3), padding=(1))
        self.bn2_1 = nn.BatchNorm2d(64)
        self.conv2_2 = nn.Conv2d(64, 64, (3, 3), padding=1)
        self.bn2_2 = nn.BatchNorm2d(64)

        self.conv2_3 = nn.Conv2d(64, 64, (3, 3), padding=(1))
        self.bn2_3 = nn.BatchNorm2d(64)
        self.conv2_4 = nn.Conv2d(64, 64, (3, 3), padding=1)
        self.bn2_4 = nn.BatchNorm2d(64)

        self.conv2_5 = nn.Conv2d(64, 64, (3, 3), padding=(1))
        self.bn2_5 = nn.BatchNorm2d(64)
        self.conv2_6 = nn.Conv2d(64, 64, (3, 3), padding=1)
        self.bn2_6 = nn.BatchNorm2d(64)

        # second stage
        self.conv3_1 = nn.Conv2d(64, 128, (3, 3), stride=2, padding=3)
        self.bn3_1 = nn.BatchNorm2d(128)
        self.conv3_2 = nn.Conv2d(128, 128, (3, 3))
        self.bn3_2 = nn.BatchNorm2d(128)

        self.conv3_3 = nn.Conv2d(128, 128, (3, 3), padding=1)
        self.bn3_3 = nn.BatchNorm2d(128)
        self.conv3_4 = nn.Conv2d(128, 128, (3, 3), padding=1)
        self.bn3_4 = nn.BatchNorm2d(128)

        self.conv3_5 = nn.Conv2d(128, 128, (3, 3), padding=1)
        self.bn3_5 = nn.BatchNorm2d(128)
        self.conv3_6 = nn.Conv2d(128, 128, (3, 3), padding=1)
        self.bn3_6 = nn.BatchNorm2d(128)

        self.conv3_7 = nn.Conv2d(128, 128, (3, 3), padding=1)
        self.bn3_7 = nn.BatchNorm2d(128)
        self.conv3_8 = nn.Conv2d(128, 128, (3, 3), padding=1)
        self.bn3_8 = nn.BatchNorm2d(128)


        self.auxc_1 = nn.Conv2d(64, 128, kernel_size=1, stride=2)

        # third stage
        self.conv4_1 = nn.Conv2d(128, 256, (3, 3), stride=2, padding=3)
        self.bn4_1 = nn.BatchNorm2d(256)
        self.conv4_2 = nn.Conv2d(256, 256, (3, 3))
        self.bn4_2 = nn.BatchNorm2d(256)

        self.conv4_3 = nn.Conv2d(256, 256, (3, 3), padding=1)
        self.bn4_3 = nn.BatchNorm2d(256)
        self.conv4_4 = nn.Conv2d(256, 256, (3, 3), padding=1)
        self.bn4_4 = nn.BatchNorm2d(256)

        self.conv4_5 = nn.Conv2d(256, 256, (3, 3), padding=1)
        self.bn4_5 = nn.BatchNorm2d(256)
        self.conv4_6 = nn.Conv2d(256, 256, (3, 3), padding=1)
        self.bn4_6 = nn.BatchNorm2d(256)
        self.conv4_7 = nn.Conv2d(256, 256, (3, 3), padding=1)
        self.bn4_7 = nn.BatchNorm2d(256)
        self.conv4_8 = nn.Conv2d(256, 256, (3, 3), padding=1)
        self.bn4_8 = nn.BatchNorm2d(256)
        self.conv4_9 = nn.Conv2d(256, 256, (3, 3), padding=1)
        self.bn4_9 = nn.BatchNorm2d(256)
        self.conv4_10 = nn.Conv2d(256, 256, (3, 3), padding=1)
        self.bn4_10 = nn.BatchNorm2d(256)
        self.conv4_11 = nn.Conv2d(256, 256, (3, 3), padding=1)
        self.bn4_11 = nn.BatchNorm2d(256)
        self.conv4_12 = nn.Conv2d(256, 256, (3, 3), padding=1)
        self.bn4_12 = nn.BatchNorm2d(256)

        self.auxc_2 = nn.Conv2d(128, 256, kernel_size=1, stride=2)

        # fourth stage
        self.conv5_1 = nn.Conv2d(256, 512, (3, 3), padding=1, stride=2)
        self.bn5_1 = nn.BatchNorm2d(512)
        self.conv5_2 = nn.Conv2d(512, 512, (3, 3), padding=1)
        self.bn5_2 = nn.BatchNorm2d(512)

        self.conv5_3 = nn.Conv2d(512, 512, (3, 3), padding=1)
        self.bn5_3 = nn.BatchNorm2d(512)
        self.conv5_4 = nn.Conv2d(512, 512, (3, 3), padding=1)
        self.bn5_4 = nn.BatchNorm2d(512)

        self.conv5_5 = nn.Conv2d(512, 512, (3, 3), padding=1)
        self.bn5_5 = nn.BatchNorm2d(512)
        self.conv5_6 = nn.Conv2d(512, 512, (3, 3), padding=1)
        self.bn5_6 = nn.BatchNorm2d(512)
        self.auxc5 = nn.Conv2d(256, 512, kernel_size=1, stride=2)

        self.avgpool = nn.AvgPool2d((4, 16))

        self.final = nn.Linear(512, 4)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.pool1(nn.functional.relu(x))

        #starts firt blocks
        x2 = self.conv2_1(x)
        x2 = nn.functional.relu(self.bn2_1(x2))
        x2 = self.conv2_2(x2)
        x2 = nn.functional.relu(self.bn2_2(x2))
        x = nn.functional.relu(x + x2)

        x2 = self.conv2_3(x)
        x2 = nn.functional.relu(self.bn2_3(x2))
        x2 = self.conv2_4(x2)
        x2 = nn.functional.relu(self.bn2_4(x2))
        x = nn.functional.relu(x + x2)

        x2 = self.conv2_5(x)
        x2 = nn.functional.relu(self.bn2_5(x2))
        x2 = self.conv2_6(x2)
        x2 = nn.functional.relu(self.bn2_6(x2))
        x = nn.functional.relu(x + x2)

        #second stage
        x2 = self.conv3_1(x)
        x2 = nn.functional.relu(self.bn3_1(x2))
        x2 = self.conv3_2(x2)
        x2 = nn.functional.relu(self.bn3_2(x2))
        x = nn.functional.relu(self.auxc_1(x) + x2)

        x2 = self.conv3_3(x)
        x2 = nn.functional.relu(self.bn3_3(x2))
        x2 = self.conv3_4(x2)
        x2 = nn.functional.relu(self.bn3_4(x2))
        x = nn.functional.relu(x + x2)

        x2 = self.conv3_5(x)
        x2 = nn.functional.relu(self.bn3_5(x2))
        x2 = self.conv3_6(x2)
        x2 = nn.functional.relu(self.bn3_6(x2))
        x = nn.functional.relu(x + x2)
        x2 = self.conv3_7(x)
        x2 = nn.functional.relu(self.bn3_7(x2))
        x2 = self.conv3_8(x2)
        x2 = nn.functional.relu(self.bn3_8(x2))
        x = nn.functional.relu(x + x2)


        #thjrd stage
        x2 = self.conv4_1(x)
        x2 = nn.functional.relu(self.bn4_1(x2))
        x2 = self.conv4_2(x2)
        x2 = nn.functional.relu(self.bn4_2(x2))
        x = nn.functional.relu(self.auxc_2(x) + x2)

        x2 = self.conv4_3(x)
        x2 = nn.functional.relu(self.bn4_3(x2))
        x2 = self.conv4_4(x2)
        x2 = nn.functional.relu(self.bn4_4(x2))
        x = nn.functional.relu(x + x2)

        x2 = self.conv4_5(x)
        x2 = nn.functional.relu(self.bn4_5(x2))
        x2 = self.conv4_6(x2)
        x2 = nn.functional.relu(self.bn4_6(x2))
        x = nn.functional.relu(x + x2)

        x2 = self.conv4_7(x)
        x2 = nn.functional.relu(self.bn4_7(x2))
        x2 = self.conv4_8(x2)
        x2 = nn.functional.relu(self.bn4_8(x2))
        x = nn.functional.relu(x + x2)

        x2 = self.conv4_9(x)
        x2 = nn.functional.relu(self.bn4_9(x2))
        x2 = self.conv4_10(x2)
        x2 = nn.functional.relu(self.bn4_10(x2))
        x = nn.functional.relu(x + x2)

        x2 = self.conv4_11(x)
        x2 = nn.functional.relu(self.bn4_11(x2))
        x2 = self.conv4_12(x2)
        x2 = nn.functional.relu(self.bn4_12(x2))
        x = nn.functional.relu(x + x2)

        #fourth stage

        x2 = self.conv5_1(x)
        x2 = nn.functional.relu(self.bn5_1(x2))
        x2 = self.conv5_2(x2)
        x2 = nn.functional.relu(self.bn5_2(x2))
        x = self.auxc5(x)
        x = nn.functional.relu(x + x2)
        x2 = self.conv5_3(x)
        x2 = nn.functional.relu(self.bn5_3(x2))
        x2 = self.conv5_4(x2)
        x2 = nn.functional.relu(self.bn5_4(x2))
        x = nn.functional.relu(x + x2)

        x2 = self.conv5_5(x)
        x2 = nn.functional.relu(self.bn5_5(x2))
        x2 = self.conv5_6(x2)
        x2 = nn.functional.relu(self.bn5_6(x2))
        x = nn.functional.relu(x + x2)

        x = self.avgpool(x)

        x = self.final(x.view(-1, 512))
        return x.squeeze()

class MLP(nn.Module):

    def __init__(self, tcontext, fft_size, dropout):
        print('Using a MLP')
        self.tcontext = tcontext
        self.fft_size = fft_size
        self.dropout=dropout
        super(MLP, self).__init__()

        self.drop1 = nn.Dropout(p=0.5)
        self.drop2 = nn.Dropout(p=0.5)
        self.drop3 = nn.Dropout(p=0.5)


        self.fc1 = nn.Linear(self.tcontext*int(self.fft_size/2+1), 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 4)

    def forward(self, x):

        x = x.view(-1, self.tcontext*int(self.fft_size/2+1))
        x = nn.functional.relu(self.fc1(x))
        if self.dropout:
            x=self.drop1(x)
        x = nn.functional.relu(self.fc2(x))
        if self.dropout:
            x=self.drop2(x)
        x = self.fc3(x)

        return x.squeeze()

class DCNN(nn.Module):
    def __init__(self, tcontext, fft_size, dropout):
        print('Using a D-CNN')
        self.tcontext = tcontext
        self.fft_size = fft_size
        self.dropout=dropout
        super(DCNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 64, (5, 5), stride=(1,4))
        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.conv2 = nn.Conv2d(64, 64, (5, 5))
        self.pool2 = nn.MaxPool2d(2, stride=2)
        self.conv3 = nn.Conv2d(64, 64, (5, 5))
        self.pool3 = nn.MaxPool2d(2, stride=2)
        self.conv4 = nn.Conv2d(64, 64, (5, 5))
        self.pool4 = nn.MaxPool2d(2, stride=2)
        self.conv5 = nn.Conv2d(64, 64, (3, 3))
        self.pool5 = nn.MaxPool2d(2, stride=2)


        self.drop1 = nn.Dropout2d(p=0.5)



        self.fc1 = nn.Linear(64, 4)


    def forward(self, x):
        x = self.conv1(x)
        if self.dropout:
            x = self.drop1(x)

        x = self.pool1(x)

        x = self.conv2(x)

        if self.dropout:
            x = self.drop1(x)
        x = self.pool2(x)

        x = self.conv3(x)
        if self.dropout:
            x = self.drop1(x)
        x = self.pool3(x)
        x = self.conv4(x)
        if self.dropout:
            x = self.drop1(x)
        x = self.pool4(x)
        x = self.conv5(x)
        if self.dropout:
            x = self.drop1(x)
        x = self.pool5(x)
        x = x.view(-1, 64)
        x = nn.functional.relu(self.fc1(x))
        if self.dropout:
            x = self.drop1(x)
        return x.squeeze()
