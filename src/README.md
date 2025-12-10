
## Data Information
### GNPC
Cohort?
Serum, Plasma, CSF separate?

Somalogic01V1_anonymized.csv -- Somalogic12V1_anonymized.csv 
    - each csv has 33848 * 650~ entries
    - 33848 rows each contains one proteomic assays
    - columns records different aptamer 
    - total number of aptamer columns across csv is 7745
    - EDTA Plasma: 
      - Somascan 7k (7595 -- 7596): 19655 -- 19249
      - Somascan 5k (5315): 2254
      - Somascan 3k (4013): 996
      - Somascan 1k (1317): 95
    - Serum:
      - Somascan 7k (7486): 4212
    - CSF:
      - Somascan 7k (7596): 1955
      - Somascan 5k (5284): 122
      - Somascan 3k (4784 -- 4785): 1156 -- 1155
    - Citrate Plasma:
      - Somascan 7k (7596): 638

SomalogicAnalyteInfoV1_anonymized.csv
    - 7644 * 14 entries
    - 7644 rows each contains aptamer
    - 14 columns records the informataion about each aptamer like target, gene information and etc
    - there are 101 aptamer that are in the data missing in this reference csv
    - Missing aptamer: {'seq_14222_68', 'seq_10367_26', 'seq_6096_23', 'seq_2626_3', 'seq_9731_29', 'seq_12926_118', 'seq_14089_41', 'seq_10450_3', 'seq_8487_62', 'seq_10696_217', 'seq_8362_102', 'seq_8816_44', 'seq_9777_138', 'seq_8484_8', 'seq_14704_10', 'seq_9466_43', 'seq_7696_104', 'seq_3737_6', 'seq_11244_63', 'seq_4475_60', 'seq_5766_6', 'seq_5647_51', 'seq_14169_66', 'seq_5059_8', 'seq_9304_27', 'seq_12908_15', 'seq_5071_3', 'seq_3743_1', 'seq_8370_102', 'seq_11835_8', 'seq_2455_17', 'seq_14582_57', 'seq_6240_55', 'seq_3031_66', 'seq_6601_1', 'seq_10009_2', 'seq_9560_56', 'seq_13592_22', 'seq_5095_7', 'seq_5118_74', 'seq_6634_4', 'seq_5247_17', 'seq_6973_2', 'seq_10492_51', 'seq_10735_12', 'seq_8683_119', 'seq_7220_20', 'seq_5073_30', 'seq_11108_16', 'seq_9921_14', 'seq_10387_1', 'seq_8765_23', 'seq_13400_13', 'seq_9382_110', 'seq_11124_9', 'seq_11505_1', 'seq_9307_3', 'seq_11170_9', 'seq_10795_32', 'seq_4494_63', 'seq_8379_35', 'seq_8644_101', 'seq_8827_1', 'seq_8829_4', 'seq_14272_43', 'seq_6956_37', 'seq_10422_44', 'seq_10872_103', 'seq_3700_15', 'seq_13718_1', 'seq_3590_8', 'seq_8256_57', 'seq_11267_11', 'seq_6643_62', 'seq_3893_55', 'seq_6650_20', 'seq_4593_11', 'seq_2795_23', 'seq_10635_33', 'seq_8339_72', 'seq_4695_49', 'seq_11259_71', 'seq_4550_3', 'seq_11276_1', 'seq_8387_33', 'seq_10901_334', 'seq_2914_65', 'seq_6297_49', 'seq_2292_17', 'seq_7093_20', 'seq_9356_20', 'seq_13428_52', 'seq_6484_11', 'seq_10689_5', 'seq_9189_76', 'seq_7195_12', 'seq_14643_27', 'seq_8798_29', 'seq_13716_259', 'seq_10878_1', 'seq_14495_161'}

SomalogicMetaV1_anonymized.csv
    - 33848 * 21 entries
    - 33848 rows each contains one proteomic assays
    - 21 columns records the somalogic setting of each patient sample
    - data contains 23 different cohort
    - units column cohort O CSF data is log2 RFU, all other cohort and sample matrix are all -1, which means cohort O CSF data has been taken log2 while others have not, need attention.

ClinicalV1_anonymized.csv
    - 31083 * 54 entries
    - 31083 rows each contains one peripheral plasma, serum or CSF samples,
    - 54 columns are different clinical information of one samples (age, race, etc)
    - 7371 samples have at least 2 visit
    - 3373 samples have at least 3 visit
    - 1051 samples have at least 4 visit
    - 125 samples have at least 5 visit
    - 2 samples have 6 visit
  - Update:
    - 3956 patients have at least 2 visit
    - 2318 patients have at least 3 visit
    - 926 patients have at least 4 visit
    - 122 patients have at least 5 visit
    - 2 patients have at least 6 visit
  - Q:
    - why the mismatch

PersonMappingV1_anonymized.csv
    - 31083 * 4 entries
    - 31083 rows each contains one peripheral plasma, serum or CSF samples,
    - 4 columns indicate whether one sample uses somalogic and mass spec

GeneticsV1_anonymized.csv
    - 22263 * 4 entries

MassSpecV1_anonymized.csv
    - 983 * 883 entries
    - maybe each row is a patient sample and each column is a protein mass spec value?

GNPCLookupV1_anonymized.csv
    - 205 * 4 entries
    - for columns that has encoded values, map it to its real value 

Sample to person mapping table for longitudinal samples data dictionary.csv
    - 59 rows * 10 columns
    - encoded value to real value mapping for longitudinal related patient features

Disease-specific genotyping data dictionary.csv
    - 59 * 10 entries

Clinical data dictionary.csv
    - 59 * 10 entries



  





