import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

try:
    from ReadData import ReadData
    from EDA import EDA
    from Prepare import Prepare
    from Train import Train
except ModuleNotFoundError:
    print("Please insert the module files in the program directory.")

def test():
    
    try:
        file = ReadData('activity_context_tracking_data.csv')
    #When the filename in incorrect or file not in place
    except IOError as ioerr:
        print("File error: " + str(ioerr))
        print("Check the filename and put it in the program directory.")
    #When filename is correct but contents are not
    except ValueError:
        print("Invalid value in a field")
    #When contents of file are manipulated
    except NameError as ne:
        print("Variable cannot be found", ne)
    #Any other fault
    except:
        print("Unknown error happened")
    else:
        data_raw = file.loadfile()
        data_eda = EDA(data_raw)
        data_encode = data_eda.encode(data_raw)#.iloc[:10000,:]

        #Create menu
        while True:
            print("-------------------------------------------------------------------------")
            print("[1]  Descriptive Statistics")
            print("[2]  Exploratory Data Analysis")
            print("[3]  Data Preparation")
            print("[4]  Train the Model")
            print("[0]  EXIT")
            print("-------------------------------------------------------------------------")
            try:
                option1 = int(input("Choose one of the options: ").strip())

                #Descriptive Statistics
                if option1 == 1:
                    try:
                        print("-------------------------------------------------------------------------")
                        print("DESCRIPTIVE STATISTICS:")
                        print("[1]  Compute Means")
                        print("[2]  Compute Medians")
                        print("[3]  Compute Standard Deviations")
                        print("[4]  Compute Variances")
                        print("[5]  Compute Minimums")
                        print("[6]  Compute Maximums")
                        print("[7]  Compute Skewnesses")
                        print("[8]  Compute Kurtosis")
                        print("-------------------------------------------------------------------------")
                        try:
                            option2 = int(input("Choose one of the descriptive statistics options: ").strip())
                            
                            #Mean
                            if option2 == 1:
                                try:
                                    print("-------------------------------------------------------------------------")
                                    for column, mean in data_eda.mean().items():
                                        print(f"Mean of {column}:\t {mean}")
                                    print("-------------------------------------------------------------------------")
                                #For values not in data dictionary
                                except KeyError:
                                    print("NO SUCH DATA.")
                                    input("Press Enter to continue...")
                        
                            #Median
                            elif option2 == 2:
                                try:
                                    print("-------------------------------------------------------------------------")
                                    for column, mean in data_eda.median().items():
                                        print(f"Median of {column}:\t {mean}")
                                    print("-------------------------------------------------------------------------")
                                #For values not in data dictionary
                                except KeyError:
                                    print("NO SUCH DATA.")
                                    input("Press Enter to continue...")
                        
                            #Standard Deviation
                            elif option2 == 3:
                                try:
                                    print("-------------------------------------------------------------------------")
                                    for column, mean in data_eda.stddev().items():
                                        print(f"Standard Deviation of {column}:\t {mean}")
                                    print("-------------------------------------------------------------------------")
                                #For values not in data dictionary
                                except KeyError:
                                    print("NO SUCH DATA.")
                                    input("Press Enter to continue...")

                            #Variance
                            elif option2 == 4:
                                try:
                                    print("-------------------------------------------------------------------------")
                                    for column, mean in data_eda.var().items():
                                        print(f"Variance of {column}:\t {mean}")
                                    print("-------------------------------------------------------------------------")
                                #For values not in data dictionary
                                except KeyError:
                                    print("NO SUCH DATA.")
                                    input("Press Enter to continue...")

                            #Minimum
                            elif option2 == 5:
                                try:
                                    print("-------------------------------------------------------------------------")
                                    for column, mean in data_eda.min().items():
                                        print(f"Minimum of {column}:\t {mean}")
                                    print("-------------------------------------------------------------------------")
                                #For values not in data dictionary
                                except KeyError:
                                    print("NO SUCH DATA.")
                                    input("Press Enter to continue...")

                            #Maximum
                            elif option2 == 6:
                                try:
                                    print("-------------------------------------------------------------------------")
                                    for column, mean in data_eda.max().items():
                                        print(f"Maximum of {column}:\t {mean}")
                                    print("-------------------------------------------------------------------------")
                                #For values not in data dictionary
                                except KeyError:
                                    print("NO SUCH DATA.")
                                    input("Press Enter to continue...")

                            #Skewness
                            elif option2 == 7:
                                try:
                                    print("-------------------------------------------------------------------------")
                                    for column, mean in data_eda.skew().items():
                                        print(f"Skewness of {column}:\t {mean}")
                                    print("-------------------------------------------------------------------------")
                                #For values not in data dictionary
                                except KeyError:
                                    print("NO SUCH DATA.")
                                    input("Press Enter to continue...")

                            #Kutrtosis
                            elif option2 == 8:
                                try:
                                    print("-------------------------------------------------------------------------")
                                    for column, mean in data_eda.kurt().items():
                                        print(f"Kurtosis of {column}:\t {mean}")
                                    print("-------------------------------------------------------------------------")
                                #For values not in data dictionary
                                except KeyError:
                                    print("NO SUCH DATA.")
                                    input("Press Enter to continue...")
                                
                            #Option not in menu
                            else:
                                print("INVALID OPTION.")
                                input("Press Enter to continue...")
                                
                        #Non-numeric option
                        except ValueError:
                            print("INVALID INPUT.")
                            input("Press Enter to continue...")
                        
                    #Non-numeric option
                    except ValueError:
                        print("INVALID INPUT.")
                        input("Press Enter to continue...")
                                
                #Exploratory Data Analysis
                elif option1 == 2:
                    try:
                        print("-------------------------------------------------------------------------")
                        print("EXPLORATORY DATA ANALYSIS:")
                        print("[1]  Explore Data")
                        print("[2]  Variables Info")
                        print("[3]  Dataset Shape")
                        print("[4]  Number of Unique Values of Each Feature")
                        print("[5]  Number of Missing Values in Each Feature")
                        print("[6]  Class Distribution in Target Variable")
                        print("[7]  Distributions of Features")
                        print("[8]  Correlation Matrix")
                        print("[9]  Correlation Scatters")
                        print("-------------------------------------------------------------------------")
                        try:
                            option2 = int(input("Choose one of the EDA options: ").strip())
                            
                            #Explore
                            if option2 == 1:
                                try:
                                    print("-------------------------------------------------------------------------")
                                    print("Top five records:")
                                    display(data_raw.head(5))
                                    print("\n\n\nBottom five records:")
                                    display(data_raw.tail(5))
                                    print("-------------------------------------------------------------------------")
                                #For values not in data dictionary
                                except KeyError:
                                    print("NO SUCH DATA.")
                                    input("Press Enter to continue...")
                                    
                            #Info
                            elif option2 == 2:
                                try:
                                    print("-------------------------------------------------------------------------")
                                    print("Variables' types and info:\n")
                                    display(data_raw.info())
                                    print("-------------------------------------------------------------------------")
                                #For values not in data dictionary
                                except KeyError:
                                    print("NO SUCH DATA.")
                                    input("Press Enter to continue...")
                                    
                            #Shape
                            elif option2 == 3:
                                try:
                                    print("-------------------------------------------------------------------------")
                                    print("Dataset shape:")
                                    display(data_raw.shape)
                                    print("-------------------------------------------------------------------------")
                                #For values not in data dictionary
                                except KeyError:
                                    print("NO SUCH DATA.")
                                    input("Press Enter to continue...")
                                    
                            #Unique
                            elif option2 == 4:
                                try:
                                    print("-------------------------------------------------------------------------")
                                    print("Number of Unique Values of Each Feature:\n")
                                    display(data_raw.nunique())
                                    print("-------------------------------------------------------------------------")
                                #For values not in data dictionary
                                except KeyError:
                                    print("NO SUCH DATA.")
                                    input("Press Enter to continue...")
                                    
                            #Missing
                            elif option2 == 5:
                                try:
                                    print("-------------------------------------------------------------------------")
                                    print("Number of Missing Values in Each Feature: \n")
                                    display(data_raw.isna().sum())
                                    print("-------------------------------------------------------------------------")
                                #For values not in data dictionary
                                except KeyError:
                                    print("NO SUCH DATA.")
                                    input("Press Enter to continue...")
                                    
                            #Value counts
                            elif option2 == 6:
                                try:
                                    print("-------------------------------------------------------------------------")
                                    print("Frequency of Classes in Target Variable: \n")
                                    data_eda.valcount()
                                    print("-------------------------------------------------------------------------")
                                #For values not in data dictionary
                                except KeyError:
                                    print("NO SUCH DATA.")
                                    input("Press Enter to continue...")
                            
                            #Distribution
                            elif option2 == 7:
                                try:
                                    print("-------------------------------------------------------------------------")
                                    print("Distributions of Features: \n")
                                    data_eda.dist()
                                    print("-------------------------------------------------------------------------")
                                #For values not in data dictionary
                                except KeyError:
                                    print("NO SUCH DATA.")
                                    input("Press Enter to continue...")
                            
                            #Correlation Matrix
                            elif option2 == 8:
                                try:
                                    print("-------------------------------------------------------------------------")
                                    print("Correlations of Features: \n(TAKES A FEW MINUTES)\n")
                                    data_eda.correlation()
                                    print("-------------------------------------------------------------------------")
                                #For values not in data dictionary
                                except KeyError:
                                    print("NO SUCH DATA.")
                                    input("Press Enter to continue...")
                                    
                            #Correlation Scatters
                            elif option2 == 9:
                                try:
                                    print("-------------------------------------------------------------------------")
                                    print("Correlations Scatterplots: \n(TAKES A FEW MINUTES)\n")
                                    data_eda.scatter()
                                    print("-------------------------------------------------------------------------")
                                #For values not in data dictionary
                                except KeyError:
                                    print("NO SUCH DATA.")
                                    input("Press Enter to continue...")
        
                            #Option not in menu
                            else:
                                print("INVALID OPTION.")
                                input("Press Enter to continue...")
                                
                        #Non-numeric option
                        except ValueError:
                            print("INVALID INPUT.")
                            input("Press Enter to continue...")
                        
                    #Non-numeric option
                    except ValueError:
                        print("INVALID INPUT.")
                        input("Press Enter to continue...")
                        
                #Data preparation
                elif option1 == 3:
                    try:
                        print("-------------------------------------------------------------------------")
                        print("DATA PREPARATION:")
                        print(">>>Each step applies to the output of last executed step<<<")
                        print("[1]  Deduplication")
                        print("[2]  Class Merging")
                        print("[3]  Class Balancing")
                        print("[4]  Feature Selection (ANOVA)")
                        print("[5]  Feature Extraction (PCA) (APPLIES ONLY TO BALANCED DATA)")
                        print("[6]  Split Train/Test")
                        print("[7]  Standardization (APPLIES ONLY TO SPLIT DATA)")
                        print("[8]  Normalization (APPLIES ONLY TO SPLIT DATA OR STANDARDIZED DATA)")
                        print("-------------------------------------------------------------------------")
                        try:
                            option2 = int(input("Choose one of the data preparation options: ").strip())
                            
                            #Deduplication
                            if option2 == 1:
                                try:
                                    print("-------------------------------------------------------------------------")
                                    data_prepare = Prepare(data_encode)
                                    data_dedup = data_prepare.Deduplicate()
                                                                   
                                    print("\nShape of data before Deduplication:   ", data_encode.shape)
                                    print("\nShape of data after Deduplication:    ", data_dedup.shape)
                                    print("\n-------------------------------------------------------------------------")
                                #For values not in data dictionary
                                except KeyError:
                                    print("NO SUCH DATA.")
                                    input("Press Enter to continue...")
                                    
                            #Merge
                            elif option2 == 2:
                                try:
                                    print("-------------------------------------------------------------------------")
                                    comp = int(input("Enter frequency threshold for merging classes (In Percent):"))
                                    try:
                                        data_prepare = Prepare(data_dedup, comp)
                                        before = data_dedup.shape
                                    except:
                                        data_prepare = Prepare(data_encode, comp)
                                        before = data_encode.shape
                                    
                                    data_merge = data_prepare.Merge()
                                    
                                    large_values = data_merge[0]['activity'].value_counts().values
                                    large_labels = data_merge[0]['activity'].value_counts().index
                                    small_values = data_merge[1]['activity'].value_counts().values
                                    small_labels = data_merge[1]['activity'].value_counts().index
                                    plt.pie(large_values, labels=large_labels, autopct='%1.1f%%', textprops={'fontsize': 6})
                                    plt.title('Distribution of classes after merging')
                                    plt.show()
                                    plt.pie(small_values, labels=small_labels, autopct='%1.1f%%', textprops={'fontsize': 6})
                                    plt.title('Distribution of small merged classes')
                                    plt.show()          
                                    print("\n-------------------------------------------------------------------------")
                                #For values not in data dictionary
                                except KeyError as k:
                                    print("NO SUCH DATA.\n", k)
                                    input("Press Enter to continue...")
                                    
                            #Balancing
                            elif option2 == 3:
                                try:
                                    print("-------------------------------------------------------------------------")
                                    comp = int(input("1.Oversampling   2.Undersampling   (1/2):"))
                                    try:
                                        data_prepare = Prepare(data_merge[0], comp)
                                        before = data_merge[0].shape
                                    except:
                                        try:
                                            data_prepare = Prepare(data_dedup, comp)
                                            before = data_dedup.shape
                                        except:
                                            data_prepare = Prepare(data_encode, comp)
                                            before = data_encode.shape
                                    
                                    data_balance = data_prepare.Balance()
                                    
                                    values = data_balance['activity'].value_counts().values
                                    labels = data_balance['activity'].value_counts().index
                                    plt.pie(values, labels=labels, autopct='%1.1f%%', textprops={'fontsize': 6})
                                    plt.show()
                                    
                                    print("\n\nShape of data before Class Balancing:   ", before)
                                    print("Shape of data after Class Balancing:    ", data_balance.shape)                                    
                                    print("\n-------------------------------------------------------------------------")
                                #For values not in data dictionary
                                except KeyError:
                                    print("NO SUCH DATA.")
                                    input("Press Enter to continue...")
                              
                            #Feature selection
                            elif option2 == 4:
                                try:
                                    print("-------------------------------------------------------------------------")
                                    comp = int(input("Enter number of features for Feature Selection:"))
                                    try:
                                        data_prepare = Prepare(data_balance, comp)
                                        before = data_balance.shape
                                    except:
                                        try:
                                            data_prepare = Prepare(data_merge, comp)
                                            before = data_merge.shape
                                        except:
                                            try:
                                                data_prepare = Prepare(data_dedup, comp)
                                                before = data_dedup.shape
                                            except:
                                                data_prepare = Prepare(data_encode, comp)
                                                before = data_encode.shape
                                        
                                    data_feature = pd.concat([data_prepare.FeatSelect()[0],
                                                            data_prepare.FeatSelect()[1]], axis = 1)
                                    
                                    print("\n\nSelected Features:")
                                    display(data_prepare.FeatSelect()[2])                                
                                    print("\n\nTop 5 rows before Feature Selection:")
                                    display(data_encode.head(5))
                                    print("\n\nTop 5 rows after Feature Selection:")
                                    display(data_feature.head(5))
                                    print("\n\nShape of data before Feature Selection:   ", before)
                                    print("Shape of data after Feature Selection:    ", data_feature.shape)
                                    print("\n-------------------------------------------------------------------------")
                                #For values not in data dictionary
                                except KeyError:
                                    print("NO SUCH DATA.")
                                    input("Press Enter to continue...")
                                    
                            #PCA
                            elif option2 == 5:
                                try:
                                    print("-------------------------------------------------------------------------")
                                    comp = int(input("Enter number of components for PCA:"))
                                    try:
                                        data_prepare = Prepare(data_feature, comp)
                                        before = data_feature.shape
                                    except:
                                        try:
                                            data_prepare = Prepare(data_balance, comp)
                                            before = data_balance.shape
                                        except:
                                            print("\nBALANCED DATA NOT FOUND.")
                                            print("PLEASE START OVER.")
                                            break
                                    
                                    data_pca = pd.concat([data_prepare.PCA2()[0], data_prepare.PCA2()[1]], axis = 1)
                                    print("\n\nTop 5 rows before PCA:")
                                    display(data_encode.head(5))
                                    print("\n\nTop 5 rows after PCA:")
                                    display(data_pca.head(5))
                                    print("\n\nShape of data before PCA:   ", before)
                                    print("Shape of data after PCA:    ", data_pca.shape)
                                    print("\n-------------------------------------------------------------------------")
                                #For values not in data dictionary
                                except KeyError:
                                    print("NO SUCH DATA.")
                                    input("Press Enter to continue...")
                                    
                            #TrainTest
                            elif option2 == 6:
                                try:
                                    print("-------------------------------------------------------------------------")
                                    comp = int(input("Enter percent of test data (0-100):"))/100
                                    try:
                                        data_prepare = Prepare(data_pca, comp)
                                        before = data_pca.shape
                                    except:
                                        try:
                                            data_prepare = Prepare(data_feature, comp)
                                            before = data_feature.shape
                                        except:
                                            try:
                                                data_prepare = Prepare(data_balance, comp)
                                                before = data_balance.shape
                                            except:
                                                try:
                                                    data_prepare = Prepare(data_merge, comp)
                                                    before = data_merge.shape
                                                except:
                                                    try:
                                                        data_prepare = Prepare(data_dedup)
                                                        before = data_dedup.shape
                                                    except:
                                                        data_prepare = Prepare(data_encode, comp)
                                                        before = data_encode.shape
                                    
                                    X_Train = data_prepare.Split()[0]
                                    X_Test = data_prepare.Split()[1]
                                    Y_Train = data_prepare.Split()[2]
                                    Y_Test = data_prepare.Split()[3]
                                    print("\n\nShape of data before Split:\n               ", before)
                                    print("\nShape of data after Split:")
                                    print("    X Train:   ", X_Train.shape)
                                    print("    X Test:    ", X_Test.shape)
                                    print("    Y Train:   ", Y_Train.shape)
                                    print("    Y Test:    ", Y_Test.shape)
                                    print("\n-------------------------------------------------------------------------")
                                #For values not in data dictionary
                                except KeyError:
                                    print("NO SUCH DATA.")
                                    input("Press Enter to continue...")
                                    
                            #Standardization
                            elif option2 == 7:
                                try:
                                    print("-------------------------------------------------------------------------")
                                    try:
                                        data_prepare = Prepare(X_Train, X_Test)
                                    except:
                                        print("\nSPLIT DATA NOT FOUND.")
                                        print("RESTART THE PROGRAM.")
                                        break
                                        
                                    X_Train_S = data_prepare.Standard()[0]
                                    X_Test_S = data_prepare.Standard()[1]
                                    print("\n\nTop 5 rows of features before Standardization:")
                                    display(X_Train.head(5))
                                    print("\n\nTop 5 rows of features after Standardization:")
                                    display(pd.DataFrame(X_Train_S).head())
                                    print("\n-------------------------------------------------------------------------")
                                #For values not in data dictionary
                                except KeyError:
                                    print("NO SUCH DATA.")
                                    input("Press Enter to continue...")
                                    
                            #Normalization
                            elif option2 == 8:
                                try:
                                    print("-------------------------------------------------------------------------")
                                    try:
                                        data_prepare = Prepare(X_Train_S, X_Test_S)
                                        before = X_Train_S
                                    except:
                                        try:
                                            data_prepare = Prepare(X_Train, X_Test)
                                            before = X_Train
                                        except:
                                            print("\nSPLIT DATA NOT FOUND.")
                                            print("PLEASE START OVER.")
                                            break
                                        
                                    X_Train_SN = data_prepare.Normal()[0]
                                    X_Test_SN = data_prepare.Normal()[1]
                                    print("\n\nTop 5 rows of features before Normalization:")
                                    display(pd.DataFrame(before).head(5))
                                    print("\n\nTop 5 rows of features after Normalization:")
                                    display(pd.DataFrame(X_Train_SN).head())
                                    print("\n-------------------------------------------------------------------------")
                                #For values not in data dictionary
                                except KeyError:
                                    print("NO SUCH DATA.")
                                    input("Press Enter to continue...")
                                    
                            #Option not in menu
                            else:
                                print("INVALID OPTION.")
                                input("Press Enter to continue...")
                                
                        #Non-numeric option
                        except ValueError as v:
                            print("INVALID INPUT.\n", v)
                            input("Press Enter to continue...")
                        
                    #Non-numeric option
                    except ValueError:
                        print("INVALID INPUT.")
                        input("Press Enter to continue...")
                        
                #Train
                elif option1 == 4:
                    try:
                        print("-------------------------------------------------------------------------")
                        print("TRAIN THE MODEL:")
                        print("[1]  Random Forest")
                        print("[2]  Support Vector Machine")
                        print("[3]  Multi Layer Perceptron")
                        print("[4]  K-Nearest Neighbours")
                        print("-------------------------------------------------------------------------")
                        try:
                            option2 = int(input("Choose one of the train options: ").strip())
                            
                            #Random forest
                            if option2 == 1:
                                try:
                                    print("-------------------------------------------------------------------------")
                                    print("RANDOM FOREST:")
                                    comp = int(input("Enter number of estimators:"))
                                    try:
                                        Xtrn = X_Train_SN
                                        Xtst = X_Test_SN
                                        Ytrn = Y_Train
                                        Ytst = Y_Test
                                    except:
                                        try:
                                            Xtrn = X_Train_S
                                            Xtst = X_Test_S
                                            Ytrn = Y_Train
                                            Ytst = Y_Test
                                        except:
                                            try:
                                                Xtrn = X_Train
                                                Xtst = X_Test
                                                Ytrn = Y_Train
                                                Ytst = Y_Test
                                            except:
                                                print("\nSPLIT DATA NOT FOUND.")
                                                print("PLEASE START OVER.")
                                                break
                                    
                                    train = Train(Xtrn, Xtst, Ytrn, Ytst, comp)
                                    print("\nPlease wait for the model to be trained...")
                                    start_time = time.time()
                                    cm = pd.DataFrame(train.Forest()[0])
                                    cr = train.Forest()[1]
                                    actr = train.SVM()[2]
                                    acts = train.SVM()[3]
                                    #cv = train.SVM()[4]
                                    train.Evaluate(cm, cr, actr, acts)#, cv)
                                    end_time = time.time()
                                    running_time = round(end_time - start_time, 0)
                                    print("\nRF training time:", running_time, "seconds")
                                    print("-------------------------------------------------------------------------")
                                #For values not in data dictionary
                                except KeyError:
                                    print("NO SUCH DATA.")
                                    input("Press Enter to continue...")
                                    
                            #SVM
                            elif option2 == 2:
                                try:
                                    print("-------------------------------------------------------------------------")
                                    print("SUPPORT VECTOR MACHINE:")
                                    comp = int(input("Enter C parameter:"))
                                    try:
                                        Xtrn = X_Train_SN
                                        Xtst = X_Test_SN
                                        Ytrn = Y_Train
                                        Ytst = Y_Test
                                    except:
                                        try:
                                            Xtrn = X_Train_S
                                            Xtst = X_Test_S
                                            Ytrn = Y_Train
                                            Ytst = Y_Test
                                        except:
                                            try:
                                                Xtrn = X_Train
                                                Xtst = X_Test
                                                Ytrn = Y_Train
                                                Ytst = Y_Test
                                            except:
                                                print("\nSPLIT DATA NOT FOUND.")
                                                print("PLEASE START OVER.")
                                                break
                                    
                                    train = Train(Xtrn, Xtst, Ytrn, Ytst, comp)
                                    print("\nPlease wait for the model to be trained...")
                                    start_time = time.time()
                                    cm = pd.DataFrame(train.SVM()[0])
                                    cr = train.SVM()[1]
                                    actr = train.SVM()[2]
                                    acts = train.SVM()[3]
                                    #cv = train.SVM()[4]
                                    train.Evaluate(cm, cr, actr, acts)#, cv)
                                    end_time = time.time()
                                    running_time = round(end_time - start_time, 0)
                                    print("\nSVM training time:", running_time, "seconds")
                                    print("-------------------------------------------------------------------------")
                                #For values not in data dictionary
                                except KeyError:
                                    print("NO SUCH DATA.")
                                    input("Press Enter to continue...")
                                    
                            #MLP
                            elif option2 == 3:
                                try:
                                    print("-------------------------------------------------------------------------")
                                    print("MULTI LAYER PERCEPTRON:")
                                    comp = int(input("Enter number of iterations:"))
                                    try:
                                        Xtrn = X_Train_SN
                                        Xtst = X_Test_SN
                                        Ytrn = Y_Train
                                        Ytst = Y_Test
                                    except:
                                        try:
                                            Xtrn = X_Train_S
                                            Xtst = X_Test_S
                                            Ytrn = Y_Train
                                            Ytst = Y_Test
                                        except:
                                            try:
                                                Xtrn = X_Train
                                                Xtst = X_Test
                                                Ytrn = Y_Train
                                                Ytst = Y_Test
                                            except:
                                                print("\nSPLIT DATA NOT FOUND.")
                                                print("PLEASE START OVER.")
                                                break
                                    
                                    train = Train(Xtrn, Xtst, Ytrn, Ytst, comp)
                                    print("\nPlease wait for the model to be trained...")
                                    start_time = time.time()
                                    cm = pd.DataFrame(train.MLP()[0])
                                    cr = train.MLP()[1]
                                    actr = train.MLP()[2]
                                    acts = train.MLP()[3]
                                    #cv = train.MLP()[4]
                                    train.Evaluate(cm, cr, actr, acts)#, cv)
                                    end_time = time.time()
                                    running_time = round(end_time - start_time, 0)
                                    print("\nMLP training time:", running_time, "seconds")
                                    print("-------------------------------------------------------------------------")
                                #For values not in data dictionary
                                except KeyError:
                                    print("NO SUCH DATA.")
                                    input("Press Enter to continue...")
                                    
                            #KNN
                            elif option2 == 4:
                                try:
                                    print("-------------------------------------------------------------------------")
                                    print("K-NEAREST NEIGHBOURS:")
                                    comp = int(input("Enter number of neighbours:"))
                                    try:
                                        Xtrn = X_Train_SN
                                        Xtst = X_Test_SN
                                        Ytrn = Y_Train
                                        Ytst = Y_Test
                                    except:
                                        try:
                                            Xtrn = X_Train_S
                                            Xtst = X_Test_S
                                            Ytrn = Y_Train
                                            Ytst = Y_Test
                                        except:
                                            try:
                                                Xtrn = X_Train
                                                Xtst = X_Test
                                                Ytrn = Y_Train
                                                Ytst = Y_Test
                                            except:
                                                print("\nSPLIT DATA NOT FOUND.")
                                                print("PLEASE START OVER.")
                                                break
                                    
                                    train = Train(Xtrn, Xtst, Ytrn, Ytst, comp)
                                    print("\nPlease wait for the model to be trained...")
                                    start_time = time.time()
                                    cm = pd.DataFrame(train.KNN()[0])
                                    cr = train.KNN()[1]
                                    actr = train.KNN()[2]
                                    acts = train.KNN()[3]
                                    #cv = train.KNN()[4]
                                    train.Evaluate(cm, cr, actr, acts)#, cv)
                                    end_time = time.time()
                                    running_time = round(end_time - start_time, 0)
                                    print("\nKNN training time:", running_time, "seconds")
                                    print("-------------------------------------------------------------------------")
                                #For values not in data dictionary
                                except KeyError:
                                    print("NO SUCH DATA.")
                                    input("Press Enter to continue...")
                                    
                            #Option not in menu
                            else:
                                print("INVALID OPTION.")
                                input("Press Enter to continue...")
                                
                        #Non-numeric option
                        except ValueError as v:
                            print("INVALID INPUT.\n", v)
                            input("Press Enter to continue...")
                        
                    #Non-numeric option
                    except ValueError:
                        print("INVALID INPUT.")
                        input("Press Enter to continue...")
                    
                #Exit while loop and program
                elif option1 == 0:
                    print("GOODBYE.")
                    break

                #Option not in menu
                else:
                    print("INVALID OPTION.")
                    input("Press Enter to continue...")
            #Non-numeric option
            except ValueError:
                print("INVALID INPUT.")
                input("Press Enter to continue...")