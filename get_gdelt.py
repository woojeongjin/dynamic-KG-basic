import gdelt
import numpy as np
import pandas as pd

gd2 = gdelt.gdelt(version=2)

results = gd2.Search(['2018 01 01', '2018 01 31'],table='events',coverage=True)

actor1 = results['Actor1Name'].values
actor2 = results['Actor2Name'].values
event = results['EventCode'].values
eventbase = results['EventBaseCode'].values
eventroot = results['EventRootCode'].values
tim = results['DATEADDED'].values

idx = np.where(~pd.isnull(actor1))
actor1 = actor1[idx]
actor2 = actor2[idx]
event = event[idx]
eventbase = eventbase[idx]
eventroot = eventroot[idx]
tim = tim[idx]

idx = np.where(~pd.isnull(actor2))
actor1 = actor1[idx]
actor2 = actor2[idx]
event = event[idx]
eventbase = eventbase[idx]
eventroot = eventroot[idx]
tim = tim[idx]

idx = np.where(actor1 != actor2)
actor1 = actor1[idx]
actor2 = actor2[idx]
event = event[idx]
eventbase = eventbase[idx]
eventroot = eventroot[idx]
tim = tim[idx]

idx = np.argsort(tim)
actor1 = actor1[idx]
actor2 = actor2[idx]
event = event[idx]
eventbase = eventbase[idx]
eventroot = eventroot[idx]
tim = tim[idx]
print(len(actor1))
dictItems = list()
init_tim = 0
total = 0
with open("GDELT/201801.txt", "w") as f:
    # f.write("subject\tobject\trelation\trelation_base\trelation_root\ttime\n")
    f.write("subject\tobject\trelation\ttime\n")
    for i in range(len(actor1)):
        if init_tim != tim[i]:
            dictItems = list()
        # if actor1[i]+"\t"+actor2[i]+"\t"+event[i]+"\t"+eventbase[i]+"\t"+eventroot[i]+"\t"+np.array2string(tim[i]) in dictItems:
        #     continue
        # else:
        #     total += 1
        #     dictItems.append(actor1[i]+"\t"+actor2[i]+"\t"+event[i]+"\t"+eventbase[i]+"\t"+eventroot[i]+"\t"+np.array2string(tim[i]))
        #     f.write(actor1[i]+"\t"+actor2[i]+"\t"+event[i]+"\t"+eventbase[i]+"\t"+eventroot[i]+"\t"+np.array2string(tim[i])+"\n")

        if actor1[i]+"\t"+actor2[i]+"\t"+event[i]+"\t"+np.array2string(tim[i]) in dictItems:
            continue
        else:
            total += 1
            dictItems.append(actor1[i]+"\t"+actor2[i]+"\t"+event[i]+"\t"+np.array2string(tim[i]))
            f.write(actor1[i]+"\t"+actor2[i]+"\t"+event[i]+"\t"+np.array2string(tim[i])+"\n")
        init_tim = tim[i]


print(total)