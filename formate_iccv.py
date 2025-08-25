result_file = 'ICCVreproduce/testaug3__noaigc_epoch1_iter600_noval_addpos/results/20250628_024702_result.txt'
format_file = result_file.replace('.txt', '_format.txt')
with open(result_file, 'r') as f:
    result_file = f.readlines()

with open(format_file, 'w') as f:
    for line in result_file:
        data = line.strip().split(' ')
        new_line = f'{data[-1]} {float(data[0]):6f}\n'
        f.write(new_line)
         