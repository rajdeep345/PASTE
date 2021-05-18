import re
def createdata(p1,p2,p3,p4):
	f1 = open(p1,"r")
	f2 = open(p2,"w")
	f3 = open(p3,"w")
	f4 = open(p4,"w")
	for line in f1:
		s = line.split("####")
		words = s[0].split()
		sent = s[0]+'\n'
		f2.write(sent)
		tups = re.findall(r'\((.*?)\)',s[1])
		aop = ''
		ptr = ''		
		for t in tups:
			t1 = t.split("],")
			l1 = t1[0][1:].split(',')
			l2 = t1[1][2:].split(',')
			aspects =  " ".join([words[int(k)] for k in l1])
			opinions = " ".join([words[int(k)] for k in l2])
			polar = t1[2][-4:-1]
			aop += (' '+aspects + ' ; ')
			aop += (opinions + ' ; ')
			aop += (polar + ' |')
			ptr += str(int(l1[0]))+' '+str(int(l1[-1]))+' '+str(int(l2[0]))+' '+str(int(l2[-1]))+' '+polar+' | '
		aop = aop[1:-2]+'\n'
		ptr = ptr[:-2]+'\n'
		f3.write(aop)
		f4.write(ptr)
	f1.close()
	f2.close()
	f3.close()
	f4.close()
	return
def main():
    createdata("train_triplets.txt","train.sent","train.tup","train.pointer")
    createdata("test_triplets.txt","test.sent","test.tup","test.pointer")
    createdata("dev_triplets.txt","dev.sent","dev.tup","dev.pointer")


if __name__=="__main__":
    main()