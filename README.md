# Proiect_Clasificarea-terenurilor-din-imagini-satelitare

Proiectul urmărește clasificarea automată a unei imagini satelitare în una dintre
cele 10 categorii de teren din EuroSAT. Se va implementa în Python/PyTorch o rețea
neuronală convoluțională (CNN) compactă, cu trei blocuri convoluționale. Modelul va
atribui o etichetă întregii imagini, fără segmentare sau detectarea obiectelor.
Pentru augmentarea generativă, se va antrena un autoencoder variațional
condițional (CVAE) exclusiv pe datele de antrenare. Imaginile sintetice vor fi verificate
vizual și adăugate acestui subset. Se va reantrena clasificatorul de la zero, cu aceeași
arhitectură, pentru a evalua dacă datele generate ajută recunoașterea terenurilor.
