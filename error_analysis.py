# Code to check the prediction performance of individual elements of a triplet.

import os

def is_full_match(triplet, triplets):
	asp = triplet.split(';')[0].strip()
	opin = triplet.split(';')[1].strip()
	senti = triplet.split(';')[2].strip()
	for t in triplets:
		aspect = t.split(';')[0].strip()
		opinion = t.split(';')[1].strip()
		sentiment = t.split(';')[2].strip()
		if asp == aspect and opin == opinion and senti == sentiment:
			return True
	
	return False

def print_scores(gt_pos, pred_pos, correct_pos):
	print('GT Triple Count:', gt_pos, '\tPRED Triple Count:', pred_pos, '\tCORRECT Triple Count:', correct_pos)
	test_p = float(correct_pos) / (pred_pos + 1e-8)
	test_r = float(correct_pos) / (gt_pos + 1e-8)
	test_acc = (2 * test_p * test_r) / (test_p + test_r + 1e-8)
	print('Test P:', round(test_p, 3))
	print('Test R:', round(test_r, 3))
	print('Test F1:', round(test_acc, 3))

gt_ap = 0
predicted_ap = 0
correct_ap = 0
gt_ap_list = []
predicted_ap_list = []
correct_ap_list = []

gt_op = 0
predicted_op = 0
correct_op = 0
gt_op_list = []
predicted_op_list = []
correct_op_list = []

proper_positions = 0
proper_senti_count = 0

gt_pos = 0
pred_pos = 0
correct_pos = 0

with open('test.out', 'r') as f_in:
	lines = f_in.readlines()
lineCount = len(lines)

start = 1
while start < lineCount:
	desired_lines = lines[start:start+2]
	exp = desired_lines[0].strip()[9:].strip()
	gt_triplets = exp.split('|')
	exp_ap = []
	exp_op = []
	for triplet in gt_triplets:
		exp_ap.append(triplet.split(';')[0].strip())
		exp_op.append(triplet.split(';')[1].strip())
	exp_ap = set(exp_ap)
	gt_ap += len(exp_ap)
	exp_op = set(exp_op)
	gt_op += len(exp_op)
	gt_ap_list.append(len(exp_ap))
	gt_op_list.append(len(exp_op))
	gt_pos += len(gt_triplets)

	pred = desired_lines[1].strip()[10:].strip()
	if pred != '':
		pred_triplets = pred.split('|')
		pred_ap = []
		pred_op = []		 
		for triplet in pred_triplets:
			pred_ap.append(triplet.split(';')[0].strip())
			pred_op.append(triplet.split(';')[1].strip())		
		
		pred_ap = set(pred_ap)		
		predicted_ap += len(pred_ap)
		correct_ap += len(exp_ap.intersection(pred_ap))
		predicted_ap_list.append(len(pred_ap))
		correct_ap_list.append(len(exp_ap.intersection(pred_ap)))
		
		pred_op = set(pred_op)		
		predicted_op += len(pred_op)
		correct_op += len(exp_op.intersection(pred_op))		
		predicted_op_list.append(len(pred_op))
		correct_op_list.append(len(exp_op.intersection(pred_op)))

		for gt_triplet in gt_triplets:
			gt_asp = gt_triplet.split(';')[0].strip()
			gt_opin = gt_triplet.split(';')[1].strip()
			gt_senti = gt_triplet.split(';')[2].strip()
			for pred_triplet in pred_triplets:
				pred_asp = pred_triplet.split(';')[0].strip()
				pred_opin = pred_triplet.split(';')[1].strip()
				pred_senti = pred_triplet.split(';')[2].strip()
				if gt_asp == pred_asp and gt_opin == pred_opin:
					proper_positions += 1
					if gt_senti == pred_senti:
						proper_senti_count += 1

		pred_pos += len(pred_triplets)		
		for gt_triplet in gt_triplets:
			if is_full_match(gt_triplet, pred_triplets):
				correct_pos += 1	
	start += 4

p_ap = float(correct_ap) / (predicted_ap + 1e-8)
r_ap = float(correct_ap) / (gt_ap + 1e-8)
f1_ap = (2 * p_ap * r_ap) / (p_ap + r_ap + 1e-8)
print('Aspect Prediction:')
print(f'Precision: {round(p_ap,3)}')
print(f'Recall: {round(r_ap,3)}')
print(f'F1: {round(f1_ap,3)}')
p_ap = float(sum(correct_ap_list)) / (sum(predicted_ap_list) + 1e-8)
r_ap = float(sum(correct_ap_list)) / (sum(gt_ap_list) + 1e-8)
f1_ap = (2 * p_ap * r_ap) / (p_ap + r_ap + 1e-8)
print('After rechecking:')
print(f'Precision: {round(p_ap,3)}')
print(f'Recall: {round(r_ap,3)}')
print(f'F1: {round(f1_ap,3)}')
print("\n")

p_op = float(correct_op) / (predicted_op + 1e-8)
r_op = float(correct_op) / (gt_op + 1e-8)
f1_op = (2 * p_op * r_op) / (p_op + r_op + 1e-8)
print('Opinion Prediction:')
print(f'Precision: {round(p_op,3)}')
print(f'Recall: {round(r_op,3)}')
print(f'F1: {round(f1_op,3)}')
p_op = float(sum(correct_op_list)) / (sum(predicted_op_list) + 1e-8)
r_op = float(sum(correct_op_list)) / (sum(gt_op_list) + 1e-8)
f1_op = (2 * p_op * r_op) / (p_op + r_op + 1e-8)
print('After rechecking:')
print(f'Precision: {round(p_op,3)}')
print(f'Recall: {round(r_op,3)}')
print(f'F1: {round(f1_op,3)}')
print("\n")

print(f'Proper Positions: {proper_positions}')
print(f'Proper Sentiments: {proper_senti_count}')
print(f'Sentiment Prediction Accuracy: {float(proper_senti_count/proper_positions)}')
print('\n')

print_scores(gt_pos, pred_pos, correct_pos)
print('\n')

print(exp_ap)
print(pred_ap)
print(gt_ap_list[-1], predicted_ap_list[-1], correct_ap_list[-1])
print(exp_op)
print(pred_op)
print(gt_op_list[-1], predicted_op_list[-1], correct_op_list[-1])