import datetime


def preprocess():
    entity_dict = {}
    relation_dict = {}

    fw1 = open("dataset/GDELT/train.txt", "w")
    fw2 = open("dataset/GDELT/test.txt", "w")
    fw3 = open("dataset/GDELT/stat.txt", "w")

    start_time = "201801010000"

    count = 0

    format = "%Y%m%d%H%M%S"
    sub_id = 0
    ob_id = 1
    rel_id = 2
    time_id = 3



    with open("dataset/GDELT_201801.txt") as fp:
        for i,line in enumerate(fp):
            # skip first line
            count += 1
            if count == 1:
                continue
            info = line.strip().split("\t")
        
            if info[sub_id] == info[ob_id]:
                continue

            entity1_id = None
            if info[sub_id] in entity_dict:
                entity1_id = entity_dict[info[sub_id]]
            else:
                entity1_id = len(entity_dict)
                entity_dict[info[sub_id]] = entity1_id

            entity2_id = None
            if info[ob_id] in entity_dict:
                entity2_id = entity_dict[info[ob_id]]
            else:
                entity2_id = len(entity_dict)
                entity_dict[info[ob_id]] = entity2_id

            relation_id = None
            if info[rel_id] in relation_dict:
                relation_id = relation_dict[info[rel_id]]
            else:
                relation_id = len(relation_dict)
                relation_dict[info[rel_id]] = relation_id

            delta = datetime.datetime.strptime(info[time_id], format) - datetime.datetime.strptime(start_time, format)
            timestamp = int(delta.days) * 24 * 60 + int(delta.seconds / 60)
            
            if timestamp < 20 * 24 * 60:
                fw1.write("%-5d\t%-5d\t%-3d\t%-3d\t0\n" % (entity1_id, relation_id, entity2_id, timestamp))
            else:
                fw2.write("%-5d\t%-5d\t%-3d\t%-3d\t0\n" % (entity1_id, relation_id, entity2_id, timestamp))

    print(count)

    fw3.write(str(len(entity_dict)) + "\t" + str(len(relation_dict))+"\t0")
    fw1.close()
    fw2.close()
    fw3.close()

if __name__ == "__main__":
    preprocess()