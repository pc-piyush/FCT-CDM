import matplotlib.pyplot as plt

def plot_patient_similarity(distances, patient_ids, idx):
    plt.figure()
    plt.plot(distances[idx])
    plt.title(f"Similarity for Patient {patient_ids[idx]}")
    plt.xlabel("Other Patients")
    plt.ylabel("DTW Distance")
    plt.show()