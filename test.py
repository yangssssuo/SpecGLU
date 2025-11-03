from data.IRDataSet import IRDataSet

dataset = IRDataSet("data/common_rows_with_gap.csv")

with open('merged_file_rtp.csv','w') as f:
    f.write('smiles,delta_est,soc_t1,soc_t1s0,fosc_s1,homo_lumo_gap,esp,esp2\n')
    for item in dataset:
        f.write('{},{},{},{},{},{},{},{}\n'.format(
            item['smiles'], item['delta_est'], item['soc_t1'], 
            item['soc_t1s0'], item['fosc_s1'], item['homo_lumo_gap'], 
            item['esp'], item['esp2']
        ))