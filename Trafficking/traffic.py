import pandas as pd
import matplotlib.pyplot as plt
import os

if __name__ == '__main__':
    df = pd.read_csv('the_global_k_anon_dataset.csv', delimiter=';', low_memory=False)
    df.drop('A', axis=1, inplace=True)
    df.dropna(axis=0, subset=['yearOfRegistration', 'ageBroad', 'gender'], inplace=True)

    age_map = {
        '0--8': 0,
        '9--17': 1,
        '18--20': 2,
        '21--23': 3,
        '24--26': 4,
        '27--29': 5,
        '30--38': 6,
        '39--47': 7,
        '48+': 8
    }

    df['ageInt'] = [age_map[age] for age in df.loc[:, 'ageBroad']]

    males = df[df['gender'] == 'Male']
    females = df[df['gender'] == 'Female']

    r_width = 0.8
    ticks = list(range(0, len(age_map)))
    labels = age_map.keys()
    rotation = 45
    y_ticks = list(range(0, 10_001, 2_000))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    ax1.hist(males['ageInt'], color='blue', rwidth=r_width)
    ax1.set_title('Age Distribution of Trafficked Males')
    ax1.set_xticks(ticks=ticks, labels=labels, rotation=rotation)
    ax1.set_yticks(list(range(0, 10_001, 2_000)))
    ax1.set_xlabel('Age Group')
    ax1.set_ylabel('Number of Trafficked Males')

    ax2.hist(females['ageInt'], color='pink', rwidth=r_width)
    ax2.set_title('Age Distribution of Trafficked Females')
    ax2.set_xticks(ticks=ticks, labels=labels, rotation=rotation)
    ax2.set_yticks(y_ticks)
    ax2.set_xlabel('Age Group')
    ax2.set_ylabel('Number of Trafficked Females')

    fig_name = "age_distribution_of_traffic.png"

    plt.tight_layout()
    plt.savefig(fig_name, dpi=200)
    plt.close()

    os.startfile(fig_name)
