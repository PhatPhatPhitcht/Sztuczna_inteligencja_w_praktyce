import streamlit as st

st.title("Największy nagłówek - tytuł strony")
st.header("Duży nagłówek - sekcja")
st.subheader("Średni nagłówek - podsekcja")

st.text("Zwykły tekst, czcionka monospace (jak kod)")
st.write("Uniwersalna metoda - automatycznie formatuje")
st.caption("Mały, szary tekst - do podpisów i przypisów")

st.markdown("**Pogrubiony** i *kursywa*")
st.markdown("# To też nagłówek H1")
st.markdown("## Nagłówek H2")
st.markdown("""
- Lista punktowa
- Drugi element
  - Zagnieżdżony
""")
st.markdown("[Link](https://example.com)")
st.markdown("Tekst z emoji 🎉")

