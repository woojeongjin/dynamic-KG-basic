import os


class Triple(object):
	def __init__(self, head, tail, relation):
		self.s = head
		self.o = tail
		self.r = relation
		# self.t = tim

class Quadruple(object):
	def __init__(self, head, tail, relation, tim):
		self.s = head
		self.o = tail
		self.r = relation
		self.t = tim


def get_total_number(inPath, fileName):
	with open(os.path.join(inPath, fileName), 'r') as fr:
		for line in fr:
			line_split = line.split()
			return int(line_split[0]), int(line_split[1])


def load_quadruples(inPath, fileName, fileName2=None):
	with open(os.path.join(inPath, fileName), 'r') as fr:
		quadrupleList = []
		quadrupleTotal = 0
		times = set()
		for line in fr:
			quadrupleTotal += 1
			line_split = line.split()
			head = int(line_split[0])
			tail = int(line_split[2])
			rel = int(line_split[1])
			time = int(line_split[3])
			times.add(time)

			quadrupleList.append(Quadruple(head, tail, rel, time))

	if fileName2 is not None:
		assert quadrupleTotal != 0
		with open(os.path.join(inPath, fileName2), 'r') as fr:
			for line in fr:
				quadrupleTotal += 1
				line_split = line.split()
				head = int(line_split[0])
				tail = int(line_split[2])
				rel = int(line_split[1])
				time = int(line_split[3])
				times.add(time)
				quadrupleList.append(Quadruple(head, tail, rel, time))
	times = list(times)
	times.sort()
	quadrupleDict = {}
	for quadruple in quadrupleList:
		quadrupleDict[(quadruple.s, quadruple.o, quadruple.r, quadruple.t)] = True

	return quadrupleTotal, quadrupleList, quadrupleDict, times

def get_quadruple_t(quads, time):
	return [quad for quad in quads if quad.t == time]

