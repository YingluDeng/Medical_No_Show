#!/usr/bin/env python
# coding: utf-8

# 
# # Project: Investigate a Medical Appointment Dataset and Figure Out What Factors Influence patients' Scheduled Appointments Missing 
# 
# ## Table of Contents
# <ul>
# <li><a href="#intro">Introduction</a></li>
# <li><a href="#wrangling">Data Wrangling</a></li>
# <li><a href="#eda">Exploratory Data Analysis</a></li>
# <li><a href="#conclusions">Conclusions</a></li>
# </ul>

# <a id='intro'></a>
# ## Introduction
# 
# > Some Patients make an appointment and do not show up in the hospital, there are many reasons to explain this situation. This dataset was collected from 100,000 medical appointment from Brazil. By analyzing this dataframe, finding out the reasons and helping hospital to better imporve their service to patients. 

# In[915]:


#import libraries 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# <a id='wrangling'></a>
# ## Data Wrangling
# 
# > In this part, checking the cleanliness of dataset, cleaning the data and making it much easier to analyze and visualize the detailed information about each factors.
# ### General Properties

# In[916]:


# Load the data 
df = pd.read_csv('noshowappointments-kagglev2-may-2016.csv')
df.head()


# ### Analysis each associated characteristics and point out important variables
# <ul>
# <li><b>ScheduledDay</b> -- the day which patients registered the appointment</li>
# <li><b>AppointmentDay</b> -- the day which patients have to meet the doctor</li>
# <li><b>Neighbourhood</b> -- the place that the appointment takes place</li>
# <li><b>Scholarship</b> -- indicates whether the patients enroll in Brasilian welfare program or not</li>
# <li><b>Scholarship, Hipertension, Diabetes, Alcoholism, Handcap, SMS_received</b> -- 0 stands for False, 1 stands for True</li>
# </ul>

# In[917]:


df.shape


# <ul>There are total 110527 samples and 14 attributes in the dataset.</ul>

# In[918]:


df.describe()


# Generate descriptive statistics and find out some errors from the data.
# <ol>
# <li>The numerical values of patientid and appointmentid do not have any meaning for our dataset, we would drop them for later use.</li>
# <li>The minimum age shows -1, obviously it is not possible, so we consider it is an error and drop it.</li>
# <li>Handcap column is a binary varible, it should only have 1 or 0. Maximum number for it is 4, the elements contain without 1 and 0 are errors.</li>
# </ol>

# In[919]:


df.info()


# <ul>As the info shown, the data does not have any missing values. But for next step, we need to get rid of some columns we do not use including patientid and appointmentid.</ul>

# ### Data Cleaning 
# Clean the data from what we found above.

# In[920]:


df_new = df.drop(['PatientId', 'AppointmentID'], axis = 1)
df_new.head(1)


# 1. Drop the helpless columns

# In[921]:


df_new = df_new.drop(df_new.query('Age < 0').index)


# 2. Drop the error in age column

# In[922]:


df_new = df_new.drop(df_new.query('Handcap > 1').index)


# 3. Drop errors in handcap column, the true values only have 0 or 1. 

# In[923]:


df_new = df_new.rename(columns = {'No-show':'No_show'})


# 4. Change the column name from "No-show" to "No_show", because the sign "-" may be confused and cause errors.

# In[924]:


new_element = []
for ele in df_new.No_show:
    if 'No' in ele:
        new_element.append(1)
    else:
        new_element.append(0)


# In[925]:


df_new.No_show = new_element
df_new.head()


# 5. Change No_show's "No"/"Yes" to integer "1"/"0" (1 means patients showed up). Then 0 means patients did not show up.

# In[926]:


type(df_new['AppointmentDay'][0])   


# In[927]:


#change appointment day's format
df_new['AppointmentDay'] = pd.to_datetime(df_new['AppointmentDay'].astype(str), format='%Y/%m/%d')
df_new['AppointmentDay'] = pd.DatetimeIndex(df_new.AppointmentDay).normalize()
df_new['AppointmentDay']


# In[928]:


#change scheduled day's format
df_new['ScheduledDay'] = pd.to_datetime(df_new['ScheduledDay'].astype(str), format='%Y/%m/%d')
df_new['ScheduledDay'] = pd.DatetimeIndex(df_new.ScheduledDay).normalize()
df_new['ScheduledDay']


# In[929]:


#create a new column awaiting_time with the difference between appointment day and scheduled day
df_new['awaiting_time'] = (df_new['AppointmentDay'] - df_new['ScheduledDay']).dt.days
df_new.head(50)


# In[930]:


#drop the data less than 0 days
df_new = df_new[df_new['awaiting_time'] >= 0]
df_new.head(50)


# 6. Create a new column name "awaiting_time" and calculate the difference between the scheduled day and appointment day. Then drop the day less than 0 day.

# In[931]:


df_new = df_new.reset_index(drop = True)
df_new.tail(5)


# 7. Since we drop some rows, the indexs are not continuous, we need to reset the indexs.

# In[932]:


df_new.describe()


# Double check the desriptive variables if the dataset is clean.

# In[933]:


df_new.info()


# No null values. 

# In[934]:


df_new.hist(figsize = (11, 8));


# Based on the histogram, most of the patients do not have the problems of alcoholism, diabetes, handicap, hipertension. Besides, most of them do not receive scholarship. Additionally, the number of people who received SMS is as much as the people who did not receive, but the no-show people are half less than show_up people. 

# <a id='eda'></a>
# ## Exploratory Data Analysis
# 
# ### Q1: Do age affect a patient to show up in an appointment?

# In[935]:


df_new.Age[df_new.No_show == 1].mean()


# In[936]:


df_new.Age[df_new.No_show == 0].mean()


# The mean of show-up and no-show are close.

# In[937]:


df_new.Age[df_new.No_show == 1].hist(alpha = 0.5, label = 'show up')
df_new.Age[df_new.No_show == 0].hist(alpha = 0.5, label = 'no show')
plt.title("Histogram of Show-Up Situation by Various Age Group")
plt.xlabel("Age")
plt.ylabel("Population");
plt.legend();


# There are more people show up and both graphs are shewed right.

# In[1035]:


median_show = df_new.groupby('No_show').Age.median()
median_show


# The median show-up patients' age is 38. And the no-show patients' age is 33. They are close. 

# In[938]:


no_show_population = df_new.loc[df_new.No_show == 0].shape[0]
no_show_population


# In[939]:


show_up_population = df_new.loc[df_new.No_show == 1].shape[0]
show_up_population


# In[940]:


show_up_population/no_show_population


# The show-up population is four times as much as no-show people. Age is unlike to be a reason correlated with patients' missing appointments.

# ### Q2: Would patients miss their appointment because they did not receive message for reminder?

# In[941]:


df_new.groupby('SMS_received')['No_show'].value_counts()


# The number of patients who received SMS and showed up is 62389. The number of those who did not receive message and showed up is 25660. 

# In[944]:


df_new.groupby('SMS_received').No_show.mean()


# In[977]:


locations = [1, 2]
heights = df_new.groupby('SMS_received').No_show.mean()
labels = ['No SMS','Received SMS']

plt.bar(locations, heights, tick_label = labels)
plt.title('Average People Show-Up by SMS')
plt.xlabel('SMS Received or Not')
plt.ylabel('Average People Show-Up');


# The number of those who received messages is less than the part who did not receive one. Therefore, receiving SMS may not imporve patient to come in time. 

# ## Q3: Is it because the period between scheduled day and appointment day is too long so that patients forget to meet their doctor?

# In[1031]:


df_new.head()


# In[967]:


df_new.awaiting_time.describe()


# In[1006]:


#create a histogram to find the relation between awaiting time and population
df_new.awaiting_time[df_new.No_show == 1].hist(alpha = 0.5, label = 'show up', figsize = (20,5))
df_new.awaiting_time[df_new.No_show == 0].hist(alpha = 0.5, label = 'no show', figsize = (10,5))

plt.title("Histogram of Show-Up Situation by Awaiting Days")
plt.xlabel("Awaiting Days")
plt.ylabel("Population")
plt.legend();


# In[1007]:


#create a bar chart
locations = [1, 2]
heights = df_new.groupby('No_show').awaiting_time.mean()
labels = ['No Show','Show Up']

plt.bar(locations, heights, tick_label = labels)
plt.title('Average Awaiting Days by Different Attandence')
plt.xlabel('People Show Up or Not')
plt.ylabel('Average Awaiting Days');


# In[1009]:


locations = [1, 2]
heights = df_new.groupby('No_show').awaiting_time.median()
labels = ['No Show','Show Up']

plt.bar(locations, heights, tick_label = labels)
plt.title('Average Awaiting Days by Different Attandence')
plt.xlabel('People Show Up or Not')
plt.ylabel('Average Awaiting Days');


# Based on the two bar charts, one is based on mean of awaiting days, the other one is based on median awaiting days. From the descriptive statistics, the maximum awaiting days is 179, so avoiding the bias influence the result, median bar chart would be a best approach to reseach. So, for those no-show people, they need to wait for 11 days, but people only wait for 2 days in show-up group. Therefore, awaiting days affect patients' attandence, the longer time they wait, the less likely patients come in time.    

# <a id='conclusions'></a>
# ## Conclusions

# <ul>
# <li>Most new borned babies would meet their appointment in time. The median of patients' age for show-up group is 38, so patients who are around thirty-something prefer to go to hospital. Age could not be a main reason related to those missing appointments.</li>
#     
# <li>People would expect that patients who received reminder SMS would less likely to miss their appointment, but based on the analysis with question 2, we found the number of patients with no SMS is more than the opposite one. So, we could not say sending reminder SMS is a good way to remind patients, maybe there are more reasons including sending time, contents, reminder ways.</li>
# <li>From the question 3 above, patients who waited for many days might not show up in their appointments. They might forget their appointment bacause of the long-wait. Or, they found another hospital to receive treatment and forget to cancel appointment. So, improving the hospital's effectiveness and speeding up the process so that patients would show up frequently. </li>
# </ul>
