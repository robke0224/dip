import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(67)


klase_0 = np.random.uniform(0, 10, (15, 2))


klase_1 = np.random.uniform(10, 20, (15, 2))


X = np.vstack([klase_0, klase_1])
y = np.hstack([np.zeros(15), np.ones(15)])

df = pd.DataFrame({
    'x1': X[:, 0],
    'x2': X[:, 1],
    'class': y
})
df.to_csv('data_points.csv', index=False)



plt.figure(figsize=(8, 8))
plt.scatter(df[df['class'] == 0]['x1'], df[df['class'] == 0]['x2'],
            c='blue', marker='o', s=100, label='Klasė 0', edgecolors='black')
plt.scatter(df[df['class'] == 1]['x1'], df[df['class'] == 1]['x2'],
            c='red', marker='s', s=100, label='Klasė 1', edgecolors='black')
plt.xlabel('x₁')
plt.ylabel('x₂')
plt.title('Sugeneruoti duomenys')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(-0.5, 20)
plt.ylim(-0.5, 20)
plt.savefig('generated_data.png', dpi=300, bbox_inches='tight')
plt.show()






def slenkstine_funkcija(a):
    if a >= 0:
        return 1
    else:
        return 0


def sigmoidine_funkcija(a):
    return 1 / (1 + np.exp(-a))


def neuronas(x1, x2, w1, w2, b, funkcija='threshold'):

    a = w1 * x1 + w2 * x2 + b

    if funkcija == 'threshold':
        return slenkstine_funkcija(a)
    else:
        sig = sigmoidine_funkcija(a)
        if sig >= 0.5:
            return 1
        else:
            return 0



def ieskoti_svoriu(duomenys, funkcija='threshold', kiek=3):
    X = duomenys[['x1', 'x2']].values
    y = duomenys['class'].values

    rasti_sprendiniai = []
    bandymai = 0
    max_bandymai = 100000

    while len(rasti_sprendiniai) < kiek and bandymai < max_bandymai:
        w1 = np.random.uniform(-10, 10)
        w2 = np.random.uniform(-10, 10)
        b = np.random.uniform(-10, 10)

        if abs(w1) + abs(w2) < 0.000001:
            bandymai += 1
            continue

        visi_teisingi = True
        for i in range(len(X)):
            prognoze = neuronas(X[i, 0], X[i, 1], w1, w2, b, funkcija)
            if prognoze != y[i]:
                visi_teisingi = False
                break


        if visi_teisingi:
            naujas = True
            for s in rasti_sprendiniai:
                if (abs(s[0] - w1) < 0.001 and
                        abs(s[1] - w2) < 0.001 and
                        abs(s[2] - b) < 0.001):
                    naujas = False
                    break

            if naujas:
                rasti_sprendiniai.append((w1, w2, b))
                print(f"  Rastas #{len(rasti_sprendiniai)}: w1={w1:.4f}, w2={w2:.4f}, b={b:.4f}")

        bandymai += 1

    print(f"  Ieškota per {bandymai} bandymus")
    return rasti_sprendiniai


print("\nSLENKSTINĖ FUNKCIJA:")
sprendiniai_threshold = ieskoti_svoriu(df, 'threshold', 3)

print("\nSIGMOIDINĖ FUNKCIJA:")
sprendiniai_sigmoid = ieskoti_svoriu(df, 'sigmoid', 3)


def braizyti_tieses_ir_vektorius(duomenys, sprendiniai, pavadinimas, failas):

    plt.figure(figsize=(8, 8))

    plt.scatter(duomenys[duomenys['class'] == 0]['x1'],
                duomenys[duomenys['class'] == 0]['x2'],
                c='blue', marker='o', s=100, label='Klasė 0',
                edgecolors='black', alpha=0.7)
    plt.scatter(duomenys[duomenys['class'] == 1]['x1'],
                duomenys[duomenys['class'] == 1]['x2'],
                c='red', marker='s', s=100, label='Klasė 1',
                edgecolors='black', alpha=0.7)

    spalvos = ['green', 'purple', 'orange']
    x_reiksmes = np.linspace(-0.5, 20, 300)

    for i, (w1, w2, b) in enumerate(sprendiniai):
        spalva = spalvos[i]

        if abs(w2) > 0.0001:
            y_reiksmes = -(w1 / w2) * x_reiksmes - (b / w2)
            plt.plot(x_reiksmes, y_reiksmes, color=spalva, linewidth=2,
                     label=f'Tiesė #{i + 1}: {w1:.2f}x₁+{w2:.2f}x₂+{b:.2f}=0')
        else:
            x0 = -b / w1
            plt.axvline(x0, color=spalva, linewidth=2,
                        label=f'Tiesė #{i + 1}: x₁={x0:.2f}')

        vardiklis = w1 ** 2 + w2 ** 2
        px = -b * w1 / vardiklis
        py = -b * w2 / vardiklis


        ilgis = np.sqrt(w1 ** 2 + w2 ** 2)
        vx = w1 / ilgis
        vy = w2 / ilgis

        plt.arrow(px, py, vx * 1.5, vy * 1.5,
                  color=spalva, width=0.08, head_width=0.3,
                  head_length=0.2, alpha=0.9)


        plt.plot(px, py, 'o', color=spalva, markersize=8,
                 markeredgecolor='black', markeredgewidth=1)

    plt.xlim(-0.5, 20)
    plt.ylim(-0.5, 20)
    plt.xlabel('x₁')
    plt.ylabel('x₂')
    plt.title(pavadinimas)
    plt.legend(loc='upper left', fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.gca().set_aspect('equal')
    plt.savefig(failas, dpi=300, bbox_inches='tight')
    plt.show()




braizyti_tieses_ir_vektorius(df, sprendiniai_threshold,
                            'Slenkstinė funkcija',
                            'boundaries_threshold.png')


braizyti_tieses_ir_vektorius(df, sprendiniai_sigmoid,
                            'Sigmoidinė funkcija',
                            'boundaries_sigmoid.png')


