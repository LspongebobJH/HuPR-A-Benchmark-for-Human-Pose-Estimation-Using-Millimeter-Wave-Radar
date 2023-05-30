import json
import numpy as np
from PIL import Image
from plot_utils import PlotMaps, PlotHeatmaps
import os

from argparse import ArgumentParser
from tqdm import tqdm

class RadarObject():
    def __init__(self, index):
        self.index = index

        numGroup = 276
        self.root = 'HuPR'
        self.saveRoot = 'HuPR'
        self.sensorType = 'iwr1843'
        self.radarDataFileNameGroup = []
        self.saveDirNameGroup =  []
        self.rgbFileNameGroup = []
        self.jointsFileNameGroup = []
        self.numADCSamples = 256
        self.adcRatio = 4
        self.numAngleBins = self.numADCSamples//self.adcRatio
        self.numEleBins = 8
        self.numRX = 4
        #self.numTX = 2
        self.numLanes = 2
        self.framePerSecond = 10
        self.duration = 60
        self.numFrame = self.framePerSecond * self.duration
        self.numChirp = 64 * 3 
        self.idxProcChirp = 64
        self.numGroupChirp = 4
        self.numKeypoints = 14 
        self.xIndices = [-45, -30, -15, 0, 15, 30, 45]
        self.yIndices = [i * 10 for i in range(10)]
        self.initialize(numGroup)

    def initialize(self, numGroup):
        # train
        if self.index == 1:
            _list = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 18, 19, 20, 21, 22, 23, 24, 25, 26]
        elif self.index == 2:
            _list = [84, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 31, 32, 53, 54, 55, 67]
        elif self.index == 3:
            _list = [102, 103, 104, 105, 106, 107, 108, 109, 110, 119, 121, 122, 123, 124, 133, 134, 135]
        elif self.index == 4:
            _list = [199, 200, 201, 202, 203, 204, 206, 207, 208, 209, 210, 211, 212, 213, 215, 216, 223, 224]
        elif self.index == 5:
            _list = [63, 64, 66, 71, 73, 74, 75, 76, 77, 78, 79, 80, 81, 83, 28, 229, 230, 259]
        elif self.index == 6:
            _list = [44, 45, 46, 47, 48, 49, 50, 51, 52, 58, 59, 60, 61, 62, 27, 28, 29, 30, 33, 35, 36, 37, 43]
        elif self.index == 7:
            _list = [160, 167, 168, 169, 170, 177, 179, 180, 187, 188, 189, 190, 68, 69, 70, 72, 85, 86, 100, 157, 158]
        elif self.index == 8:
            _list = [153, 154, 155, 162, 163, 165, 166, 171, 172, 173, 174, 175, 138, 147, 148, 149, 150, 151, 152]
        elif self.index == 9:
            _list = [235, 236, 258, 261, 262, 265, 266, 267, 268, 271, 272, 275, 276, 225, 226, 231, 232, 233, 234]
        elif self.index == 10:
            _list = [176, 182, 183, 184, 185, 186, 191, 192, 193, 195, 196, 198, 260, 263, 264, 269, 270, 273, 274]

        # validation
        elif self.index == 11:
            _list = [1, 14, 34, 57, 65, 98]
        elif self.index == 12:
            _list = [56, 99, 159, 178, 101, 120]
        elif self.index == 13:
            _list = [137, 156, 161, 164, 181]
        elif self.index == 14:
            _list = [194, 197, 205, 257]

        self._list = _list
            
        # for i in range(1, numGroup + 1):
        for i in self._list:
            radarDataFileName = ['raw_data/' + self.sensorType + '/' + self.root + '/single_' + str(i) + '/hori', 
                                 'raw_data/' + self.sensorType + '/' + self.root + '/single_' + str(i) + '/vert']
            saveDirName = 'data/' + self.saveRoot + '/single_' + str(i)
            rgbFileName = 'raw_data/frames/' + '/single_' + str(i)
            #jointsFileName = '../data/' + self.saveRoot + '/single_' + str(i) + '/annot/hrnet_annot.json'
            self.radarDataFileNameGroup.append(radarDataFileName)
            self.saveDirNameGroup.append(saveDirName)
            self.rgbFileNameGroup.append(rgbFileName)
            #self.jointsFileNameGroup.append(jointsFileName)
    
    def postProcessFFT3D(self, dataFFT):
        dataFFT = np.fft.fftshift(dataFFT, axes=(0, 1,))
        dataFFT = np.transpose(dataFFT, (2, 0, 1))
        dataFFT = np.flip(dataFFT, axis=(1, 2))
        return dataFFT

    def getadcDataFromDCA1000(self, fileName):
        adcData = np.fromfile(fileName+'/adc_data.bin', dtype=np.int16)
        fileSize = adcData.shape[0]
        adcData = adcData.reshape(-1, self.numLanes*2).transpose()
        # # for complex data
        fileSize = int(fileSize/2)
        LVDS = np.zeros((2, fileSize))  # seperate each LVDS lane into rows

        temp = np.empty((adcData[0].size + adcData[1].size), dtype=adcData[0].dtype)
        temp[0::2] = adcData[0]
        temp[1::2] = adcData[1]
        LVDS[0] = temp
        temp = np.empty((adcData[2].size + adcData[3].size), dtype=adcData[2].dtype)
        temp[0::2] = adcData[2]
        temp[1::2] = adcData[3]
        LVDS[1] = temp

        adcData = np.zeros((self.numRX, int(fileSize/self.numRX)), dtype = 'complex_')
        iter = 0
        for i in range(0, fileSize, self.numADCSamples * 4):
            adcData[0][iter:iter+self.numADCSamples] = LVDS[0][i:i+self.numADCSamples] + np.sqrt(-1+0j)*LVDS[1][i:i+self.numADCSamples]
            adcData[1][iter:iter+self.numADCSamples] = LVDS[0][i+self.numADCSamples:i+self.numADCSamples*2] + np.sqrt(-1+0j)*LVDS[1][i+self.numADCSamples:i+self.numADCSamples*2]
            adcData[2][iter:iter+self.numADCSamples] = LVDS[0][i+self.numADCSamples*2:i+self.numADCSamples*3] + np.sqrt(-1+0j)*LVDS[1][i+self.numADCSamples*2:i+self.numADCSamples*3]
            adcData[3][iter:iter+self.numADCSamples] = LVDS[0][i+self.numADCSamples*3:i+self.numADCSamples*4] + np.sqrt(-1+0j)*LVDS[1][i+self.numADCSamples*3:i+self.numADCSamples*4]
            iter = iter + self.numADCSamples

        #correct reshape
        adcDataReshape = adcData.reshape(self.numRX, -1, self.numADCSamples)
        print('Shape of radar data:', adcDataReshape.shape)
        return adcDataReshape

    def clutterRemoval(self, input_val, axis=0):
        """Perform basic static clutter removal by removing the mean from the input_val on the specified doppler axis.
        Args:
            input_val (ndarray): Array to perform static clutter removal on. Usually applied before performing doppler FFT.
                e.g. [num_chirps, num_vx_antennas, num_samples], it is applied along the first axis.
            axis (int): Axis to calculate mean of pre-doppler.
        Returns:
            ndarray: Array with static clutter removed.
        """
        # Reorder the axes
        reordering = np.arange(len(input_val.shape))
        reordering[0] = axis
        reordering[axis] = 0
        input_val = input_val.transpose(reordering)

        # Apply static clutter removal
        mean = input_val.transpose(reordering).mean(0)
        output_val = input_val - np.expand_dims(mean, axis=0)
        out = output_val.transpose(reordering)
        return out 

    def generateHeatmap(self, frame):
        # horizontal
        dataRadar = np.zeros((self.numRX*2, self.idxProcChirp, self.numADCSamples), dtype='complex_')
        # vertical
        dataRadar2 = np.zeros((self.numRX, self.idxProcChirp, self.numADCSamples), dtype='complex_')
        
        # Process radar data with TDM-MIMO
        for idxRX in range(self.numRX):
            for idxChirp in range(self.numChirp):
                if idxChirp % 3 == 0:
                    dataRadar[idxRX, idxChirp//3] = frame[idxRX, idxChirp]
                if idxChirp % 3 == 1:
                    dataRadar2[idxRX, idxChirp//3] = frame[idxRX, idxChirp]
                elif idxChirp % 3 == 2:
                    dataRadar[idxRX+4, idxChirp//3] = frame[idxRX, idxChirp]

        # step1: clutter removal
        dataRadar = np.transpose(dataRadar, (1, 0, 2))
        dataRadar = self.clutterRemoval(dataRadar, axis=0)
        dataRadar = np.transpose(dataRadar, (1, 0, 2))
        dataRadar2 = np.transpose(dataRadar2, (1, 0, 2))
        dataRadar2 = self.clutterRemoval(dataRadar2, axis=0)
        dataRadar2 = np.transpose(dataRadar2, (1, 0, 2))

        # step2: range-doppler FFT
        for idxRX in range(self.numRX * 2):
            dataRadar[idxRX, :, :] = np.fft.fft2(dataRadar[idxRX, :, :])
        for idxRX in range(self.numRX * 1):
            dataRadar2[idxRX, :, :] = np.fft.fft2(dataRadar2[idxRX, :, :])

        # step3: angle FFT
        padding = ((0, self.numAngleBins - dataRadar.shape[0]), (0,0), (0,0))
        dataRadar = np.pad(dataRadar, padding, mode='constant')
        padding2 = ((2, self.numAngleBins - 4 - 2), (0,0), (0,0))
        dataRadar2 = np.pad(dataRadar2, padding2, mode='constant')
        dataMerge = np.stack((dataRadar, dataRadar2))
        paddingEle = ((0, self.numEleBins - dataMerge.shape[0]), (0,0), (0,0), (0,0))
        dataMerge = np.pad(dataMerge, paddingEle, mode='constant')
        for idxChirp in range(self.idxProcChirp):
            for idxADC in range(self.numADCSamples):
                dataMerge[:, 2, idxChirp, idxADC] = np.fft.fft(dataMerge[:, 2, idxChirp, idxADC])
                dataMerge[:, 3, idxChirp, idxADC] = np.fft.fft(dataMerge[:, 3, idxChirp, idxADC])
                dataMerge[:, 4, idxChirp, idxADC] = np.fft.fft(dataMerge[:, 4, idxChirp, idxADC])
                dataMerge[:, 5, idxChirp, idxADC] = np.fft.fft(dataMerge[:, 5, idxChirp, idxADC])
                for idxEle in range(self.numEleBins):
                    dataMerge[idxEle, :, idxChirp, idxADC] = np.fft.fft(dataMerge[idxEle, :, idxChirp, idxADC])

        # select specific area of ADCSamples (containing signal responses)
        idxADCSpecific = [i for i in range(94, 30, -1)] # 84, 20
        rate = self.adcRatio

        # shft the velocity information
        dataTemp = np.zeros((self.idxProcChirp, self.numADCSamples//rate, self.numAngleBins, self.numEleBins), dtype='complex_')
        dataFFTGroup = np.zeros((self.idxProcChirp//self.numGroupChirp, self.numADCSamples//rate, self.numAngleBins, self.numEleBins), dtype='complex_')
        for idxEle in range(self.numEleBins):
            for idxRX in range(self.numAngleBins):
                for idxADC in range(self.numADCSamples//rate):
                    dataTemp[:, idxADC, idxRX, idxEle] = dataMerge[idxEle, idxRX, :, idxADCSpecific[idxADC]]
                    dataTemp[:, idxADC, idxRX, idxEle] = np.fft.fftshift(dataTemp[:, idxADC, idxRX, idxEle], axes=(0))
        
        # select specific velocity information
        chirpPad = self.idxProcChirp//self.numGroupChirp
        i = 0
        for idxChirp in range(self.idxProcChirp//2 - chirpPad//2, self.idxProcChirp//2 + chirpPad//2):
            dataFFTGroup[i, :, :, :] = self.postProcessFFT3D(np.transpose(dataTemp[idxChirp, :, :, :], (1, 2, 0)))
            i += 1

        return dataFFTGroup  

    def saveDataAsFigure(self, img, joints, output, visDirName, idxFrame, output2=None):
        heatmap = PlotHeatmaps(joints, self.numKeypoints)
        PlotMaps(visDirName, self.xIndices, self.yIndices, 
        idxFrame, output, img, heatmap, output2)
    
    def saveRadarData(self, matrix, dirName, idxFrame):
        dirSave = dirName + ('/%09d' % idxFrame) + '.npy'
        np.save(dirSave, matrix)

    def processRadarDataHoriVert(self):
        #numpoints = []
        for idxName in range(len(self.radarDataFileNameGroup)):
            adcDataHori = self.getadcDataFromDCA1000(self.radarDataFileNameGroup[idxName][0])
            adcDataVert = self.getadcDataFromDCA1000(self.radarDataFileNameGroup[idxName][1])
            for idxFrame in range(0,self.numFrame):
                if not os.path.exists(os.path.join(self.saveDirNameGroup[idxName], 'hori', f'{idxFrame:09d}.npy')):
                    frameHori = adcDataHori[:, self.numChirp*(idxFrame):self.numChirp*(idxFrame+1), 0:self.numADCSamples]
                    outputHori = self.generateHeatmap(frameHori)
                    self.saveRadarData(outputHori, self.saveDirNameGroup[idxName] + '/hori', idxFrame)
                    print('index:%d, %d/%d, %s, finished frame %d hori' % 
                      (self.index, idxName, len(self.radarDataFileNameGroup), self.radarDataFileNameGroup[idxName][0], idxFrame), end='\r')
                if not os.path.exists(os.path.join(self.saveDirNameGroup[idxName], 'vert', f'{idxFrame:09d}.npy')):
                    frameVert = adcDataVert[:, self.numChirp*(idxFrame):self.numChirp*(idxFrame+1), 0:self.numADCSamples]
                    outputVert = self.generateHeatmap(frameVert)
                    self.saveRadarData(outputVert, self.saveDirNameGroup[idxName] + '/vert', idxFrame)
                    print('index:%d, %d/%d, %s, finished frame %d vert' % 
                      (self.index, idxName, len(self.radarDataFileNameGroup), self.radarDataFileNameGroup[idxName][0], idxFrame), end='\r')
                    
    def loadDataPlot(self):
        for idxName in [0, 1, 2, 3]:
            with open('data/HuPR/hrnet_annot_test.json', "r") as fp:
                annotGroup = json.load(fp)
            for idxFrame in range(0,self.numFrame):
                hori_path = self.saveDirNameGroup[idxName] + '/hori' + ('/%09d' % idxFrame) + '.npy'
                vert_path = self.saveDirNameGroup[idxName] + '/vert' + ('/%09d' % idxFrame) + '.npy'
                outputHori = np.load(hori_path)
                outputVert = np.load(vert_path)
                outputHori = np.mean(np.abs(outputHori), axis=(0, 3))
                outputVert = np.mean(np.abs(outputVert), axis=(0, 3))
                visDirName = self.saveDirNameGroup[idxName] + '/visualization' + ('/%09d.png' % idxFrame)
                img = np.array(Image.open(self.rgbFileNameGroup[idxName] + "/%09d.jpg" % idxFrame).convert('RGB'))
                joints = annotGroup[idxName][idxFrame]['joints']
                self.saveDataAsFigure(img, joints, outputHori, visDirName, idxFrame, outputVert)
                print('%s, finished frame %d' % (self.radarDataFileNameGroup[idxName][0], idxFrame), end='\r')

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--index', type=int)
    parser.add_argument('--vis', default=False, action='store_true')
     = parser.parse_args()
    index = .index
    
    visualization = .vis
    radarObject = RadarObject(index)
    if not visualization:
        radarObject.processRadarDataHoriVert()
    else:
        radarObject.loadDataPlot()
