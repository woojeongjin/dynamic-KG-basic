def make_id(num, filename):
    with open(filename, 'w') as f:
        f.write(str(num)+'\n')
        for i in range(num):
            f.write(str(i)+'\t'+str(i)+'\n')

def make_data(filename, outname):
    num_lines = 0
    with open(filename, 'r') as f:
        for line in f:
            num_lines += 1
    print(num_lines)

    with open(outname, 'w') as fw:
        fw.write(str(num_lines)+'\n')
        with open(filename, 'r') as f:
            for line in f:
                info = line.strip().split("\t")
                sub = info[0]
                rel = info[1]
                ob = info[2]
                fw.write(sub+'\t'+ob+'\t'+rel+'\n')


if __name__ == "__main__":
    path = "benchmarks"
    data_name = "/ICEWS18"
    with open(path+data_name+'/stat.txt', 'r') as f:
        for line in f:
            nums = line.strip().split('\t')
    num_ent = int(nums[0])
    num_rel = int(nums[1])
    make_id(num_ent, path+data_name+'/entity2id.txt')
    make_id(num_rel, path+data_name+'/relation2id.txt')

    make_data(path+data_name+'/train.txt', path+data_name+'/train2.txt')
    make_data(path+data_name+'/valid.txt', path+data_name+'/valid2.txt')
    make_data(path+data_name+'/test.txt', path+data_name+'/test2.txt')

