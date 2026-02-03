import numpy as np

# with open('jobs.txt', 'w') as f:
#     for bootstrap in range(200):
#         for iteration in np.arange(1, 6):
#             for shuffle in ['True', 'False']:
#                 for drop in [20,40,60,80]:
#                     f.writelines('module load miniconda ; source activate DBN; python DBN_Search_AlecSearch_AlecData.py {bootstrap} {iteration} {shuffle} {drop} \n'.format(bootstrap = bootstrap, iteration = iteration, shuffle = shuffle, drop = drop))

#     for bootstrap in range(200):
#             for shuffle in ['True', 'False']:
#                     f.writelines('module load miniconda ; source activate DBN; python DBN_Search_AlecSearch_AlecData.py {bootstrap} {iteration} {shuffle} {drop} \n'.format(bootstrap = bootstrap, iteration = 1, shuffle = shuffle, drop = 0))

                    
with open('jobs_Anirban_Data.txt', 'w') as f:
    for bootstrap in range(200):
        for iteration in [1]:
            for shuffle in ['True', 'False']:
                for drop in [0,20,40,60,80]:
                    f.writelines('module load miniconda ; source activate DBN; python DBN_Search_AnirbanSearch_AnirbanData.py {bootstrap} {iteration} {shuffle} {drop} \n'.format(bootstrap = bootstrap, iteration = iteration, shuffle = shuffle, drop = drop))



            
    
    


