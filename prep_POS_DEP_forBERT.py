import sys
import os
from transformers import BertTokenizer
from spacyface.aligner import BertAligner

def getTokenizer(mode):
	if mode == 'gen':
		tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
	elif mode == 'lap':
		print("Getting Tokenizer PT for Laptop Domain:")
		tokenizer = BertTokenizer.from_pretrained('/laptop_pt/', do_lower_case=True)
	elif mode == 'res':
		print("Getting Tokenizer PT for Restaurant Domain:")
		tokenizer = BertTokenizer.from_pretrained('/rest_pt/', do_lower_case=True)

	return tokenizer

def getAligner(mode):
	if mode == 'gen':
		alnr = BertAligner.from_pretrained("bert-base-uncased")
	elif mode == 'lap':
		print("Getting Aligner PT for Laptop Domain:")
		alnr = BertAligner.from_pretrained("/laptop_pt/")
	elif mode == 'res':
		print("Getting Aligner PT for Restaurant Domain:")
		alnr = BertAligner.from_pretrained("/rest_pt/")

	return alnr

def getPOS_DEP(sent_file, pos_out, dep_out):
	f1 = open(sent_file)
	g1 = open(pos_out,'w')
	g2 = open(dep_out,'w')

	line_count = 0
	err_count = 0
	
	for line in f1:
		line = line.strip()
		line_count += 1
		tokens = tokenizer.tokenize(line)
		features = alnr.meta_tokenize(line)
		align_tokens = [feature.token for feature in features]
		pos_tags = [feature.pos if feature.pos == 'PUNCT' else feature.pos + "-" + feature.tag for feature in features]
		dep_tags = [feature.dep for feature in features]
		if tokens == align_tokens:
			pos_tags = ' '.join(pos_tags)
			dep_tags = ' '.join(dep_tags)
			g1.write(pos_tags + '\n')
			g2.write(dep_tags + '\n')
		else:
			print(f'Counts not matching on line {line_count} in {sent_file}..')
			print(line)
			print(tokens)
			print(align_tokens)
			print("\n")
			err_count += 1

if __name__ == "__main__":
	mode = sys.argv[1]
	tokenizer = getTokenizer(mode)
	alnr = getAligner(mode)
	dirs = ['/14res/', '/15res/', '/16res/', '/resall/', '/lap14/']
	if mode == 'lap':
		dirs = [dirs[-1]]
	elif mode == 'res':
		dirs = dirs[:-1]

	for path in dirs:
		os.chdir(path)
		if mode == 'gen':
			getPOS_DEP('train.sent','trainb_pos.sent','trainb_dep.sent')
			getPOS_DEP('dev.sent','devb_pos.sent','devb_dep.sent')
			getPOS_DEP('test.sent','testb_pos.sent','testb_dep.sent')
		else:
			getPOS_DEP('train.sent','trainb_pt_pos.sent','trainb_pt_dep.sent')
			getPOS_DEP('dev.sent','devb_pt_pos.sent','devb_pt_dep.sent')
			getPOS_DEP('test.sent','testb_pt_pos.sent','testb_pt_dep.sent')
