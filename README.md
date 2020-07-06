# Medical_No_Show
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "\n",
    "# Project: Investigate a Medical Appointment Dataset and Figure Out What Factors Influence patients' Scheduled Appointments Missing \n",
    "\n",
    "## Table of Contents\n",
    "<ul>\n",
    "<li><a href=\"#intro\">Introduction</a></li>\n",
    "<li><a href=\"#wrangling\">Data Wrangling</a></li>\n",
    "<li><a href=\"#eda\">Exploratory Data Analysis</a></li>\n",
    "<li><a href=\"#conclusions\">Conclusions</a></li>\n",
    "</ul>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<a id='intro'></a>\n",
    "## Introduction\n",
    "\n",
    "> Some Patients make an appointment and do not show up in the hospital, there are many reasons to explain this situation. This dataset was collected from 100,000 medical appointment from Brazil. By analyzing this dataframe, finding out the reasons and helping hospital to better imporve their service to patients. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 915,
   "metadata": {},
   "outputs": [],
   "source": [
    "#import libraries \n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "%matplotlib inline"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<a id='wrangling'></a>\n",
    "## Data Wrangling\n",
    "\n",
    "> In this part, checking the cleanliness of dataset, cleaning the data and making it much easier to analyze and visualize the detailed information about each factors.\n",
    "### General Properties"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 916,
   "metadata": {
    "scrolled": false
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>PatientId</th>\n",
       "      <th>AppointmentID</th>\n",
       "      <th>Gender</th>\n",
       "      <th>ScheduledDay</th>\n",
       "      <th>AppointmentDay</th>\n",
       "      <th>Age</th>\n",
       "      <th>Neighbourhood</th>\n",
       "      <th>Scholarship</th>\n",
       "      <th>Hipertension</th>\n",
       "      <th>Diabetes</th>\n",
       "      <th>Alcoholism</th>\n",
       "      <th>Handcap</th>\n",
       "      <th>SMS_received</th>\n",
       "      <th>No-show</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>2.987250e+13</td>\n",
       "      <td>5642903</td>\n",
       "      <td>F</td>\n",
       "      <td>2016-04-29T18:38:08Z</td>\n",
       "      <td>2016-04-29T00:00:00Z</td>\n",
       "      <td>62</td>\n",
       "      <td>JARDIM DA PENHA</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>No</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>5.589978e+14</td>\n",
       "      <td>5642503</td>\n",
       "      <td>M</td>\n",
       "      <td>2016-04-29T16:08:27Z</td>\n",
       "      <td>2016-04-29T00:00:00Z</td>\n",
       "      <td>56</td>\n",
       "      <td>JARDIM DA PENHA</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>No</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>4.262962e+12</td>\n",
       "      <td>5642549</td>\n",
       "      <td>F</td>\n",
       "      <td>2016-04-29T16:19:04Z</td>\n",
       "      <td>2016-04-29T00:00:00Z</td>\n",
       "      <td>62</td>\n",
       "      <td>MATA DA PRAIA</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>No</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>8.679512e+11</td>\n",
       "      <td>5642828</td>\n",
       "      <td>F</td>\n",
       "      <td>2016-04-29T17:29:31Z</td>\n",
       "      <td>2016-04-29T00:00:00Z</td>\n",
       "      <td>8</td>\n",
       "      <td>PONTAL DE CAMBURI</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>No</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>8.841186e+12</td>\n",
       "      <td>5642494</td>\n",
       "      <td>F</td>\n",
       "      <td>2016-04-29T16:07:23Z</td>\n",
       "      <td>2016-04-29T00:00:00Z</td>\n",
       "      <td>56</td>\n",
       "      <td>JARDIM DA PENHA</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>No</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "      PatientId  AppointmentID Gender          ScheduledDay  \\\n",
       "0  2.987250e+13        5642903      F  2016-04-29T18:38:08Z   \n",
       "1  5.589978e+14        5642503      M  2016-04-29T16:08:27Z   \n",
       "2  4.262962e+12        5642549      F  2016-04-29T16:19:04Z   \n",
       "3  8.679512e+11        5642828      F  2016-04-29T17:29:31Z   \n",
       "4  8.841186e+12        5642494      F  2016-04-29T16:07:23Z   \n",
       "\n",
       "         AppointmentDay  Age      Neighbourhood  Scholarship  Hipertension  \\\n",
       "0  2016-04-29T00:00:00Z   62    JARDIM DA PENHA            0             1   \n",
       "1  2016-04-29T00:00:00Z   56    JARDIM DA PENHA            0             0   \n",
       "2  2016-04-29T00:00:00Z   62      MATA DA PRAIA            0             0   \n",
       "3  2016-04-29T00:00:00Z    8  PONTAL DE CAMBURI            0             0   \n",
       "4  2016-04-29T00:00:00Z   56    JARDIM DA PENHA            0             1   \n",
       "\n",
       "   Diabetes  Alcoholism  Handcap  SMS_received No-show  \n",
       "0         0           0        0             0      No  \n",
       "1         0           0        0             0      No  \n",
       "2         0           0        0             0      No  \n",
       "3         0           0        0             0      No  \n",
       "4         1           0        0             0      No  "
      ]
     },
     "execution_count": 916,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Load the data \n",
    "df = pd.read_csv('noshowappointments-kagglev2-may-2016.csv')\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Analysis each associated characteristics and point out important variables\n",
    "<ul>\n",
    "<li><b>ScheduledDay</b> -- the day which patients registered the appointment</li>\n",
    "<li><b>AppointmentDay</b> -- the day which patients have to meet the doctor</li>\n",
    "<li><b>Neighbourhood</b> -- the place that the appointment takes place</li>\n",
    "<li><b>Scholarship</b> -- indicates whether the patients enroll in Brasilian welfare program or not</li>\n",
    "<li><b>Scholarship, Hipertension, Diabetes, Alcoholism, Handcap, SMS_received</b> -- 0 stands for False, 1 stands for True</li>\n",
    "</ul>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 917,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(110527, 14)"
      ]
     },
     "execution_count": 917,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.shape"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<ul>There are total 110527 samples and 14 attributes in the dataset.</ul>"
   ]
  },
