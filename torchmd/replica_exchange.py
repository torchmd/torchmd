import numpy as np

BOLTZMAN = 0.001987191
def temp_exchange(T, Epot):
    replicas = []
    n=0
    for t, epot in zip(T, Epot):
        replicas.append([n, t, epot])
        n+=1
    replicas = np.array(replicas)
    np.random.shuffle(replicas)
    out = []
    if len(replicas)%2 !=0:
        half = int(len(replicas)/2)
        replicas_p1 = replicas[:half]
        replicas_p2 = replicas[half:-1]
        out.append(replicas[-1])
    else:
        half = int(len(replicas)/2)
        replicas_p1 = replicas[:half]
        replicas_p2 = replicas[half:]

    for n, rep_p1_i in enumerate(replicas_p1):
        rep_p2_i = replicas_p2[n]
        delta12 = ((1/(rep_p2_i[1] * BOLTZMAN)) - (1/(rep_p1_i[1] * BOLTZMAN)))*(rep_p1_i[2] - rep_p2_i[2])
        prob12 = min([1, np.exp(-delta12)])
        if prob12 > np.random.random():
            out.append([rep_p1_i[0], rep_p2_i[1], rep_p1_i[2]])
            out.append([rep_p2_i[0], rep_p1_i[1], rep_p2_i[2]])
        else:
            out.append(rep_p1_i)
            out.append(rep_p2_i)

    out = np.array(out)
    out = out[out[:,0].argsort()]
    return out[:, 1]
