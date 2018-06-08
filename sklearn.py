from sklearn.neighbors import LocalOutlierFactor
import random
#generate fake data

data=[2*i for i in range(0,200)]


outlier_data=[51*i for i in range(0,20)]
print(outlier_data)
#[0, 51, 102, 153, 204, 255, 306, 357, 408, 459, 510, 561, 612, 663, 714, 765, 816, 867, 918, 969]



mix_data = list(set(data+outlier_data))
random.shuffle(mix_data)

final_= [[i] for i in mix_data]  #making it 2 dim
clf = LocalOutlierFactor(n_neighbors=20)
y_pred = clf.fit_predict(final_)
for j,i in enumerate(y_pred):
    if i==-1:
        print(final_[j])
        
# [459]
# [612]
# [561]
# [816]
# [918]
# [396]
# [765]
# [4]
# [0]
# [714]
# [867]
# [392]
# [2]
# [969]
# [6]
# [394]
# [8]
# [510]
# [390]
# [408]
# [663]
# [398]
