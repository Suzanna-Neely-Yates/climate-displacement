## Climate Displacement

![image](https://user-images.githubusercontent.com/59842246/134459370-15d11487-f5db-4702-84e7-632e74a90dd9.png)

### Predicting Climate Migration and Refugees
Climate disasters, induced by the changing climate, are becoming increasingly prevalent. These disasters are not only causing irreparable harm to the environment, but are also creating a surge of migrants and refugees. This research seeks to create a model to predict the number of people - both climate migrants and refugees - displaced by climate disasters in a the United States.

It is difficult to train a neural network to accurately predict numbers of displaced people for two key reasons. First, there is no concrete combination or set of variables that inevitably lead to climate displacement. Second, there is no centralized data source for this information. This research will set out to solve both challenges by creating a custom dataset composed of US state-to-state migration and all climate disasters classified as emergencies by FEMA. These variables will be fed into the neural network in order to examine how the prevalence and type of natural disasters in the future may or may not impact migration patterns above typical rates. Multiple types of neural networks will be trained and validated, ultimately the most accurate model will be chosen.


This research will also include an ethical assessment. It will explore weaknesses in the dataset which may produce bias in the model - including some communities being over or under represented. It will also seek to understand the role of climate justice in the context the predictive model.

____________________________
### Related Works

In order to design a comprehensive neural network that encompasses and addresses the factors primarily causing climate migration, it is important to first examine the political and environmental climate that leads to such displacement. In a recent study, [Governing climate displacement: the ethics and politics of human resettlement](https://www.tandfonline.com/doi/full/10.1080/09644016.2012.651905), researchers found that climate change is projected to raise difficult ethical issues about what governments and international organizations should do to protect human populations displaced by climate disasters and long term environment change. When migration is planned and supported through public policy, it can stabilize and diversify livelihoods as well as reduce potential vulnerabilities to environment shocks that could result in severe consequences if not for planning. However, currently governments often actively discourage migration by using labor codes, land use restrictions, and other policy instruments in order to control movement and settlement of various populations.

Furthermore, research [Conceptualising Climate-Induced Displacement](https://www.legalanthology.ch/t/kaelin_conceptualising-climate-induced-displacement_2010.pdf) examined a current debate on the level of influence of climate change and related natural events on displacement in the world; some believe that it is quite plausibly the direct cause and factor leading the displacement, coining the term “climate induced displacement”, while others believe that it is only one of a multitude of factors of a complex problem that is displacement. Climate change can already be felt and observed today, making it crucial to take action in attempt to mitigate the impacts of natural hazards on climate induced displacement.

Research examining migration patterns, caused by climate change, have primarily been examined and modeled with direct correlations to sea level rise. A key study, <i>Modeling migration patterns in the USA under sea level rise</i>, modeled the future patterns of climate migration based on current NOAA digital coast datasets, small area population projects and a Machine Learning method for modeling human migration. After modeling the initial impacts of sea level rise, researchers used a artificial neural network (ANN) method previously fit with county migration data from the United States in order to ultimate determine climate migration predictions. The methodology and design of the models enabled researchers to produce new results that were able to differentiate between the impacts of migration, both directly and indirectly caused by sea level rise, from historical trends in migration.

Another study that attempts to model climate displacement in a different manner is <i>Drought Displacement in Kenya, Ethiopia and Somalia</i>. Researchers developed a Pastoralist Livelihood and Displacement Simulator which produced seemingly accurate estimates of displacement caused by drought throughout the horn of Africa. Unlike the previous model, this simulator includes climate, environment and social science data into a system dynamics model, a model commonly used to examine population movements and behavior of systems. However, due to data discrepancies in specific parts of the horn of Africa as well as social impacts such as changes in family structures and education, researchers recommend that the model be improved.

____________________________
### Bibliography

Craig A. Johnson (2012) Governing climate displacement: the ethics and politics of human resettlement, Environmental Politics, 21:2, 308-328, DOI: 10.1080/09644016.2012.651905

Ginnetti, Justin, and Travis Franck. “Horn of Africa Technical Report - IDMC.” Norwegian Refugee Council (NRC), Internal Displacement Monitoring Centre (IDMC), https://www.internal-displacement.org/sites/default/files/publications/documents/201405-horn-of-africa-technical-report-en.pdf 

Robinson C., Dilkina B., Moreno-Cruz J. Modeling migration patterns in the USA under sea level rise. PLoS ONE. 2020;15(1) doi: 10.1371/journal.pone.0227436

Walter Kälin, ‘Conceptualising Climate-Induced Displacement’ in Jane McAdam (ed.), Climate Change and Displacement: Multidisciplinary Perspectives (Oxford: Hart Publishing, 2010) pp. 81-103.

____________________________
### Update #1

This project will entail creating a custom dataset, utilizing two existing datasets from the IRS and Federal Emergencies and Disasters Agency (FEMA). The IRS Dataset, State-to-State US Migration Data (1990-2011) [SOI Tax Stats - Migration Data, Internal Revenue Service](https://www.irs.gov/statistics/soi-tax-stats-migration-data), contains records of the number of US Citizens moving between states which includes the attributes such as Year, State and Number of Migrants. The FEMA, Federal Emergencies and Disasters (1953-Present) [Federal Emergencies and Disasters, 1953-Present, Federal Emergency Management Agency](https://www.kaggle.com/fema/federal-disasters), dataset includes records of major natural disasters which include hurricanes, tornados, storms, high waters, wind-driven waters, tidal waves, tsunamis, earthquakes, volcanic eruptions, landslides, mudslides, snowstorms, or droughts, fires, floods, or explosions. Each natural disaster is then associated with attributes such as State, Start Date, End Date and more. The two datasets will then be merged based on mutual attributes - year and state.

We will be utilizing the Pytorch machine learning framework within Jupyter Notebook in order to execute this project. Though we will write our own code, we will be feeding our data into existing machine learning models and neural network frameworks. We will be experimenting with various neural networks to decide which will best fit our problem.

Numeric attributes from our dataset will be fed into these various models of neural networks. Ultimately, we are seeking an output similar to a standard regression problem. Our project seeks to determine the relationship between variables - migration and natural diasters - in order to predict future climate migration patterns utilizing various neural network models. The output will ultimately be the number of predicted migrants due to natural disasters.


____________________________
### Update #2

The majority of the work that we have been doing revolves around gathering and cleaning data. Our two current datasets, SOI Tax Stats - Migration Data (Internal Revenue Service and Federal Emergencies) and Disasters, 1953-Present (Federal Emergency Management Agency) have common attributes to join the tables on, however, some of the attributes do not match up. This requires us to manually match some of the states and locations to each other which has taken a fair amount of time. We are still in the process of joining the datasets.

In the meantime, we are also searching for other data sources which may provide more in-depth information about climate disasters in given smaller areas or regions. The main concerns that we have found here is that the data is incomplete or not as detailed as we would like.

Finally, our group is aiming for an A in this project. We have to complete all of the updates and tasks to the best of our ability.

____________________________
### Methods

We will be using Jupyter Notebook for this project, utilizing the Python programming language. Pytorch will be imported and ultilized in order to build our neural network. We will create multiple neural networks with the Pytorch models, analyzing how well each neural network model does with our data.

Migration Data (SQI Tax Stats) and Federal Emergency Management Agency (Disasters 1953-Present) will be joined. This joined dataset will serve as our primary dataset for this research. As previously discussed, this dataset will be fed into various Pytorch neural networks. We will analyize and explore which neural network models work best with our given dataset.
____________________________

Group: Ben Luo & Neely Yates
