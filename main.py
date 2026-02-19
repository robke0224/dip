import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Užfiksuojame atsitiktinių skaičių generatoriaus pradinę reikšmę,
# kad rezultatai būtų atkartojami kiekvieną kartą paleidus kodą
np.random.seed(67)

# Generuojame 15 taškų klasei 0 – koordinatės atsitiktinės intervale [0, 10]
klase_0 = np.random.uniform(0, 10, (15, 2))

# Generuojame 15 taškų klasei 1 – koordinatės atsitiktinės intervale [10, 20]
klase_1 = np.random.uniform(10, 20, (15, 2))

# Sujungiame abiejų klasių taškus į vieną matricą X (30 eilučių, 2 stulpeliai)
X = np.vstack([klase_0, klase_1])

# Sukuriame etikečių vektorių: 15 nulių (klasė 0) ir 15 vienetų (klasė 1)
y = np.hstack([np.zeros(15), np.ones(15)])

# Sukuriame DataFrame su stulpeliais x1, x2 (koordinatės) ir class (klasė)
df = pd.DataFrame({
    'x1': X[:, 0],
    'x2': X[:, 1],
    'class': y
})

# Išsaugome duomenis į CSV failą vėlesniam naudojimui
df.to_csv('data_points.csv', index=False)

# Vizualizuojame duomenis: klasė 0 – mėlyni apskritimai, klasė 1 – raudoni kvadratai
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


# -----------------------------------------------------------------------
# AKTYVAVIMO FUNKCIJOS
# -----------------------------------------------------------------------

def slenkstine_funkcija(a):
    """Slenkstinė (Heaviside) aktyvavimo funkcija.
    Grąžina 1, jei įėjimas a >= 0, kitu atveju – 0."""
    if a >= 0:
        return 1
    else:
        return 0


def sigmoidine_funkcija(a):
    """Sigmoidinė aktyvavimo funkcija.
    Grąžina reikšmę intervale (0, 1) – kuo didesnis a, tuo artimesnė 1."""
    return 1 / (1 + np.exp(-a))


def neuronas(x1, x2, w1, w2, b, funkcija='threshold'):
    """Vieno neurono modelis su dviem įėjimais.

    Apskaičiuoja svertinę sumą a = w1*x1 + w2*x2 + b,
    tada pritaiko pasirinktą aktyvavimo funkciją ir grąžina klasę (0 arba 1).

    Parametrai:
        x1, x2   – įėjimo reikšmės (taško koordinatės)
        w1, w2   – svoriai (kiek kiekviena koordinatė „sveria")
        b        – poslinkis (bias)
        funkcija – 'threshold' arba 'sigmoid'
    """
    # Apskaičiuojame neuro tiesinio jungiklio išėjimą (svertinė suma + poslinkis)
    a = w1 * x1 + w2 * x2 + b

    if funkcija == 'threshold':
        # Naudojame slenkstinę funkciją
        return slenkstine_funkcija(a)
    else:
        # Naudojame sigmoidinę funkciją; jei rezultatas >= 0.5 – klasė 1
        sig = sigmoidine_funkcija(a)
        if sig >= 0.5:
            return 1
        else:
            return 0


# -----------------------------------------------------------------------
# SVORIŲ PAIEŠKA ATSITIKTINIU METODU
# -----------------------------------------------------------------------

def ieskoti_svoriu(duomenys, funkcija='threshold', kiek=3):
    """Ieško neurono svorių (w1, w2, b), kurie teisingai klasifikuoja visus taškus.

    Metodas – atsitiktinė paieška: generuojami atsitiktiniai svoriai ir
    tikrinama, ar neuronas su tais svoriais klasifikuoja visus 30 taškų teisingai.
    Ieškoma tol, kol randami `kiek` skirtingų sprendinių arba pasiekiamas bandymų limitas.

    Grąžina: sąrašą kortežų (w1, w2, b).
    """
    # Išskiriame įėjimo duomenis ir etiketes
    X = duomenys[['x1', 'x2']].values
    y = duomenys['class'].values

    rasti_sprendiniai = []  # Čia kaupsiме rastus tinkamus svorių rinkinius
    bandymai = 0
    max_bandymai = 100000   # Saugiklis, kad ciklas nesitęstų be galo

    while len(rasti_sprendiniai) < kiek and bandymai < max_bandymai:
        # Generuojame atsitiktinius svorius ir poslinkį intervale [-10, 10]
        w1 = np.random.uniform(-10, 10)
        w2 = np.random.uniform(-10, 10)
        b = np.random.uniform(-10, 10)

        # Praleidžiame beveik nulinius svorius – tokie neuronai neklasifikuoja
        if abs(w1) + abs(w2) < 0.000001:
            bandymai += 1
            continue

        # Patikriname, ar šie svoriai teisingai klasifikuoja VISUS taškus
        visi_teisingi = True
        for i in range(len(X)):
            prognoze = neuronas(X[i, 0], X[i, 1], w1, w2, b, funkcija)
            if prognoze != y[i]:
                visi_teisingi = False
                break  # Pakanka vienos klaidos – šie svoriai netinka

        if visi_teisingi:
            # Patikriname, ar šis sprendinys jau nebuvo rastas anksčiau
            # (vengiame dublikatų pagal 0.001 tikslumą)
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


# Ieškome svorių naudojant slenkstinę aktyvavimo funkciją
print("\nSLENKSTINĖ FUNKCIJA:")
sprendiniai_threshold = ieskoti_svoriu(df, 'threshold', 3)

# Ieškome svorių naudojant sigmoidinę aktyvavimo funkciją
print("\nSIGMOIDINĖ FUNKCIJA:")
sprendiniai_sigmoid = ieskoti_svoriu(df, 'sigmoid', 3)


# -----------------------------------------------------------------------
# VIZUALIZACIJA: SPRENDIMO RIBOS IR NORMALIŲJŲ VEKTORIAI
# -----------------------------------------------------------------------

def braizyti_tieses_ir_vektorius(duomenys, sprendiniai, pavadinimas, failas):
    """Nubrėžia rastus sprendimo paviršius (tieses) ir jų normalius vektorius.

    Kiekvienas sprendinys w1*x1 + w2*x2 + b = 0 apibrėžia tiesę plokštumoje.
    Normalinis vektorius (w1, w2) rodo, į kurią pusę neuronas „mato" klasę 1.
    Rodyklė pradedama nuo taško, kuriame tiesė yra arčiausiai koordinačių pradžios.
    """
    plt.figure(figsize=(8, 8))

    # Pavaizduojame duomenų taškus
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

        # Braižome sprendimo tiesę w1*x1 + w2*x2 + b = 0
        if abs(w2) > 0.0001:
            # Išreiškiame x2 per x1: x2 = -(w1/w2)*x1 - b/w2
            y_reiksmes = -(w1 / w2) * x_reiksmes - (b / w2)
            plt.plot(x_reiksmes, y_reiksmes, color=spalva, linewidth=2,
                     label=f'Tiesė #{i + 1}: {w1:.2f}x₁+{w2:.2f}x₂+{b:.2f}=0')
        else:
            # Jei w2 ≈ 0, tiesė yra vertikali: x1 = -b/w1
            x0 = -b / w1
            plt.axvline(x0, color=spalva, linewidth=2,
                        label=f'Tiesė #{i + 1}: x₁={x0:.2f}')

        # Randame artimiausią tiesės tašką koordinačių pradžiai (projekcija)
        # Formulė: p = -b * (w1, w2) / (w1² + w2²)
        vardiklis = w1 ** 2 + w2 ** 2
        px = -b * w1 / vardiklis
        py = -b * w2 / vardiklis

        # Normalizuojame svorio vektorių į vienetinį ir braižome rodyklę
        ilgis = np.sqrt(w1 ** 2 + w2 ** 2)
        vx = w1 / ilgis  # Vienetinis vektorius x kryptimi
        vy = w2 / ilgis  # Vienetinis vektorius y kryptimi

        # Rodyklė rodo normalinio vektoriaus kryptį (klasės 1 pusę)
        plt.arrow(px, py, vx * 1.5, vy * 1.5,
                  color=spalva, width=0.08, head_width=0.3,
                  head_length=0.2, alpha=0.9)

        # Pažymime rodyklės pradžios tašką
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


# Vizualizuojame slenkstinės funkcijos sprendimo ribes
braizyti_tieses_ir_vektorius(df, sprendiniai_threshold,
                            'Slenkstinė funkcija',
                            'boundaries_threshold.png')

# Vizualizuojame sigmoidinės funkcijos sprendimo ribes
braizyti_tieses_ir_vektorius(df, sprendiniai_sigmoid,
                            'Sigmoidinė funkcija',
                            'boundaries_sigmoid.png')