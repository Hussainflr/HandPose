from upe.clean_data import UnifiedPoseDataset

dataset = UnifiedPoseDataset(mode='clean', loadit=False, name='train2')
print (len(dataset))
dataset = UnifiedPoseDataset(mode='test', loadit=False, name='test2')
print(len(dataset))

