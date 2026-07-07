"""Re-localize the synthetic hospital dataset to a Tunisian setting.

This script keeps the exact same graph structure (ids, dates, admission
types, room numbers, billing statistics shape, review linkage, etc.) as the
original US-based synthetic dataset, but replaces every identity/location
field with Tunisian equivalents:

- Hospitals: real Tunisian hospital/clinic names, grouped by governorate
  (instead of US state abbreviations).
- Payers: Tunisian insurers (CNAM + private insurance companies).
- Physicians: Tunisian names, Tunisian (and francophone) medical schools,
  salary rescaled to a plausible TND range.
- Patients: Tunisian names.
- Visits: unchanged except billing_amount rescaled to a plausible TND range.
- Reviews: patient/physician/hospital names recomputed from the visit they
  are linked to, so the data stays 100% internally consistent.

Run with: python scripts/generate_tunisian_dataset.py
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

# ---------------------------------------------------------------------------
# Name pools
# ---------------------------------------------------------------------------

MALE_FIRST_NAMES = [
    "Mohamed",
    "Ahmed",
    "Ali",
    "Mahdi",
    "Youssef",
    "Karim",
    "Walid",
    "Sami",
    "Anis",
    "Firas",
    "Amine",
    "Aymen",
    "Bilel",
    "Chokri",
    "Fathi",
    "Ghazi",
    "Hamza",
    "Hichem",
    "Imed",
    "Jalel",
    "Kamel",
    "Lassaad",
    "Mehdi",
    "Moez",
    "Nabil",
    "Nizar",
    "Omar",
    "Rami",
    "Riadh",
    "Sadok",
    "Skander",
    "Tarek",
    "Wael",
    "Zied",
    "Adel",
    "Bassem",
    "Chedly",
    "Elyes",
    "Farouk",
    "Habib",
    "Ismail",
    "Jamel",
    "Khalil",
    "Lotfi",
    "Marwan",
    "Othman",
    "Rachid",
    "Sofiene",
    "Taher",
    "Yassine",
    "Zoubeir",
    "Abderrahmen",
    "Bechir",
    "Chaker",
    "Driss",
    "Fares",
    "Ghassen",
    "Hedi",
    "Iheb",
    "Jawad",
    "Khaled",
    "Mongi",
    "Naceur",
    "Rafik",
    "Slim",
    "Wassim",
    "Nizar",
    "Selim",
    "Foued",
]

FEMALE_FIRST_NAMES = [
    "Amina",
    "Salma",
    "Amel",
    "Sana",
    "Rim",
    "Nour",
    "Ines",
    "Sarra",
    "Wafa",
    "Mouna",
    "Leila",
    "Emna",
    "Yosra",
    "Sonia",
    "Sihem",
    "Rania",
    "Dorra",
    "Hela",
    "Ahlem",
    "Nadia",
    "Samia",
    "Faten",
    "Meriem",
    "Khadija",
    "Aida",
    "Nesrine",
    "Olfa",
    "Manel",
    "Marwa",
    "Chaima",
    "Bouthaina",
    "Fatma",
    "Habiba",
    "Ikram",
    "Jihen",
    "Kaouther",
    "Latifa",
    "Malek",
    "Nawel",
    "Ons",
    "Radhia",
    "Safia",
    "Takwa",
    "Wided",
    "Yasmine",
    "Zeineb",
    "Asma",
    "Besma",
    "Chiraz",
    "Dhouha",
    "Eya",
    "Feriel",
    "Ghofrane",
    "Hind",
    "Imen",
    "Jamila",
    "Kalthoum",
    "Lobna",
    "Mariem",
    "Nedra",
    "Oumaima",
    "Rahma",
    "Syrine",
    "Chourouk",
]

LAST_NAMES = [
    "Ben Ali",
    "Ben Salah",
    "Trabelsi",
    "Bouazizi",
    "Gharbi",
    "Jendoubi",
    "Mabrouk",
    "Chaabane",
    "Bouzid",
    "Hamdi",
    "Belhaj",
    "Karray",
    "Sassi",
    "Ouali",
    "Rekik",
    "Ferchichi",
    "Amri",
    "Nasri",
    "Slimani",
    "Bouslama",
    "Cherif",
    "Marzouki",
    "Zoghlami",
    "Khemiri",
    "Dridi",
    "Guesmi",
    "Riahi",
    "Ayari",
    "Bahloul",
    "Kammoun",
    "Chtioui",
    "Hachani",
    "Zouari",
    "Mejri",
    "Baccouche",
    "Selmi",
    "Toumi",
    "Chouchane",
    "Jaziri",
    "Mansouri",
    "Kefi",
    "Souissi",
    "Bouden",
    "Hentati",
    "Kacem",
    "Tlili",
    "Bayoudh",
    "Frikha",
    "Naili",
    "Abidi",
    "Sfar",
    "Mnif",
    "Chabbi",
    "Charfi",
    "Hammami",
    "Jaballah",
    "Aloui",
    "Chaieb",
    "Snoussi",
    "Turki",
    "Ben Amor",
    "Ben Youssef",
    "Ben Salem",
    "Gargouri",
    "Boughanmi",
    "Msadek",
    "Guizani",
    "Ltaief",
]

# (hospital_name, governorate)
HOSPITALS = [
    ("Hopital Charles Nicolle", "Tunis"),
    ("Hopital La Rabta", "Tunis"),
    ("Hopital Habib Thameur", "Tunis"),
    ("Hopital Mongi Slim", "Tunis"),
    ("Hopital Aziza Othmana", "Tunis"),
    ("Hopital d'Enfants Bechir Hamza", "Tunis"),
    ("Hopital Militaire de Tunis", "Tunis"),
    ("Institut Salah Azaiez", "Tunis"),
    ("Clinique Hannibal", "Tunis"),
    ("Clinique Pasteur", "Tunis"),
    ("Clinique El Amen Tunis", "Tunis"),
    ("Polyclinique Les Berges du Lac", "Tunis"),
    ("Clinique Ennasr", "Tunis"),
    ("Clinique Ibn Zohr", "Tunis"),
    ("Hopital Habib Bourguiba Sfax", "Sfax"),
    ("Hopital Hedi Chaker", "Sfax"),
    ("Clinique Ennour Sfax", "Sfax"),
    ("Clinique El Amen Sfax", "Sfax"),
    ("Hopital Regional de Sfax", "Sfax"),
    ("Clinique Les Jasmins Sfax", "Sfax"),
    ("Hopital Farhat Hached", "Sousse"),
    ("Hopital Sahloul", "Sousse"),
    ("Clinique Marbella Sousse", "Sousse"),
    ("Clinique Les Oliviers Sousse", "Sousse"),
    ("Polyclinique Sousse Jawhara", "Sousse"),
    ("Clinique Erriadh Sousse", "Sousse"),
    ("Hopital Fattouma Bourguiba", "Monastir"),
    ("Clinique Regionale de Monastir", "Monastir"),
    ("Hopital Mohamed Taher Maamouri", "Nabeul"),
    ("Clinique Ennakhil Nabeul", "Nabeul"),
]

PAYERS = [
    "CNAM",
    "STAR Assurances",
    "Maghrebia Assurances",
    "GAT Assurances",
    "Assurances BIAT",
]

MEDICAL_SCHOOLS = [
    "Faculte de Medecine de Tunis",
    "Faculte de Medecine de Sfax",
    "Faculte de Medecine de Sousse",
    "Faculte de Medecine de Monastir",
    "Universite Paris Descartes - Faculte de Medecine",
    "Universite de Montreal - Faculte de Medecine",
    "Universite Laval - Faculte de Medecine",
    "Universite Libre de Bruxelles - Faculte de Medecine",
    "Universite de Strasbourg - Faculte de Medecine",
]


def random_full_names(n: int, sexes=None) -> list[str]:
    """Generate n Tunisian full names.

    If `sexes` is given (values "Male"/"Female"), the first name is drawn
    from the matching gendered pool; otherwise it's drawn from either pool.
    """
    if sexes is None:
        first_pool = np.array(MALE_FIRST_NAMES + FEMALE_FIRST_NAMES)
        first_names = RNG.choice(first_pool, size=n)
    else:
        first_names = np.empty(n, dtype=object)
        male_mask = (sexes == "Male").to_numpy()
        first_names[male_mask] = RNG.choice(MALE_FIRST_NAMES, size=male_mask.sum())
        first_names[~male_mask] = RNG.choice(
            FEMALE_FIRST_NAMES, size=(~male_mask).sum()
        )

    last_names = RNG.choice(LAST_NAMES, size=n)
    return [f"{f} {l}" for f, l in zip(first_names, last_names)]


def main() -> None:
    # -- Hospitals -----------------------------------------------------
    hospitals = pd.read_csv(DATA_DIR / "hospitals.csv")
    hospitals = hospitals.sort_values("hospital_id").reset_index(drop=True)
    assert len(hospitals) == len(HOSPITALS), "hospital count mismatch"
    hospitals["hospital_name"] = [name for name, _ in HOSPITALS]
    hospitals["hospital_state"] = [gov for _, gov in HOSPITALS]
    hospitals.to_csv(DATA_DIR / "hospitals.csv", index=False)

    # -- Payers ----------------------------------------------------------
    payers = pd.read_csv(DATA_DIR / "payers.csv")
    payers = payers.sort_values("payer_id").reset_index(drop=True)
    assert len(payers) == len(PAYERS), "payer count mismatch"
    payers["payer_name"] = PAYERS
    payers.to_csv(DATA_DIR / "payers.csv", index=False)

    # -- Physicians --------------------------------------------------------
    physicians = pd.read_csv(DATA_DIR / "physicians.csv")
    physicians["physician_name"] = random_full_names(len(physicians))
    physicians["medical_school"] = RNG.choice(MEDICAL_SCHOOLS, size=len(physicians))
    physicians["salary"] = (physicians["salary"] / 4).round(2)
    physicians.to_csv(DATA_DIR / "physicians.csv", index=False)

    # -- Patients ------------------------------------------------------
    patients = pd.read_csv(DATA_DIR / "patients.csv")
    patients["patient_name"] = random_full_names(len(patients), patients["patient_sex"])
    patients.to_csv(DATA_DIR / "patients.csv", index=False)

    # -- Visits (structure unchanged, billing rescaled to TND) -----------
    visits = pd.read_csv(DATA_DIR / "visits.csv")
    visits["billing_amount"] = (visits["billing_amount"] / 10).round(2)
    visits.to_csv(DATA_DIR / "visits.csv", index=False)

    # -- Reviews (recompute names from the linked visit) ------------------
    reviews = pd.read_csv(DATA_DIR / "reviews.csv")
    merged = (
        reviews[["review_id", "visit_id", "review"]]
        .merge(
            visits[["visit_id", "patient_id", "physician_id", "hospital_id"]],
            on="visit_id",
            how="left",
        )
        .merge(patients[["patient_id", "patient_name"]], on="patient_id", how="left")
        .merge(
            physicians[["physician_id", "physician_name"]],
            on="physician_id",
            how="left",
        )
        .merge(
            hospitals[["hospital_id", "hospital_name"]], on="hospital_id", how="left"
        )
    )
    reviews_out = merged[
        [
            "review_id",
            "visit_id",
            "review",
            "physician_name",
            "hospital_name",
            "patient_name",
        ]
    ]
    reviews_out.to_csv(DATA_DIR / "reviews.csv", index=False)

    print("Tunisian dataset generated successfully.")


if __name__ == "__main__":
    main()
