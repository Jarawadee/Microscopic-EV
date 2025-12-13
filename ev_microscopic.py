import streamlit as st
from PIL import Image
import os



def home_page():
    st.title("A platform for parasite detection")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Parasited image")

        img_1 = Image.open("sample image/TT_0660-2.tif")
        img_2 = Image.open("sample image/TT_.png")

        st.image(img_1, caption="Input image", use_container_width=True)
        st.image(img_2, caption="Detection image", use_container_width=True)

    with col2:
        st.subheader("Parasited image")

        img_3 = Image.open("sample image/TT_0661.tif")
        img_4 = Image.open("sample image/TT_2.png")

        st.image(img_3, caption="Input image", use_container_width=True)
        st.image(img_4, caption="Detection image", use_container_width=True)


def parasite_detail_page():
  
  st.write("this is page detail")
  st.sidebar.title("Table of Content")
  st.sidebar.markdown("""
  - [Ascaris lumbricoides](#ascaris-lumbricoides)
  - [Hymenolepsis nana](#hymenolepsis-nana)
  - [Hymenolepsis diminuta](#hymenolepsis-diminuta)
  - [Taenia spp.](#taenia-spp.)
  - [Trichuris trichiura](#trichuris-trichiura)
  - [Hookworm](#hookworm)
  - [Opisthorchis viverrini](#opisthorchis-viverrini)
  - [Minute intestinal flukes](#minute-intestinal-flukes)
  - [Reference](#reference)
""")
#----------------------------------------------------------------------------------------------------------------------------------------
  
#ascaris---------------------------------------------------------------------------------------------------------------------------------
  st.header("Ascaris lumbricoides", anchor="ascaris-lumbricoides")
  st.header("1.Ascaris lumbricoides")
  st.subheader("Geographical Distribution")
  st.write("""
  Ascaris lumbricoides, a common human roundworm, is found globally but is most prevalent in tropical and subtropical regions.
  """)
  st.subheader("Morphology")
  st.image('https://laboratorytests.org/wp-content/uploads/2019/03/Fertilized-and-Unfertilized-Eggs-of-Ascaris-Lumbricoides.jpg')
  st.image('https://learnzoology.wordpress.com/wp-content/uploads/2014/04/ascaris-male-female.jpg?w=640')
  st.write("""
  - **Fertilized eggs:** The fertilized eggs are laid by females after inseminated by mating with a male. These are embryonated and develop into the infective eggs.
  - **Unfertilized Eggs:** The unfertilized eggs are laid by uninseminated female. Thes are non-embryonated and cannot become infective.
  - **Adult worms:** Characterized by their large size, cylindrical, unsegmented bodies, and distinct sexual dimorphism. Females are typically 20-35 cm long, while males are slightly smaller, ranging from 15-30 cm. Both sexes have three "lips" at the anterior end of the body. 
  """)
  st.subheader("Life cycle")
  st.markdown("""
  The life cycle of *Ascaris* worms begins when adult worms reside in the small intestine of humans. In this environment, female worms—which are generally larger than males—can produce approximately 200,000 eggs per day. These eggs are passed out of the body through the stool. Only fertilized eggs are capable of causing infection. Once excreted, the fertilized eggs require specific environmental conditions to develop; they mature best in moist, warm, and shaded soil.
  People typically become infected by accidentally ingesting these eggs, often through food that has been in contact with soil contaminated by human feces containing fertilized *Ascaris* eggs. After ingestion, the eggs hatch in the small intestine, releasing larvae. These larvae then penetrate the intestinal wall and travel via the lymphatic system and bloodstream to the lungs. Within the lungs, the larvae enter the air sacs (alveoli), migrate up the respiratory tract to the throat, and are subsequently swallowed. Once back in the small intestine, the larvae mature into adult worms, completing the life cycle.
  """)
  st.image("https://edge.sitecorecloud.io/mmanual-ssq1ci05/media/home/images/a/s/c/ascaris-lumbricoides-lifecycle-lg-cdc-sized.jpg?sc_lang=en&mw=2048")
  st.markdown("""
  - **Mode of Transmission:** Ingestion of infective eggs from contaminated food or water. 
  - **Route of Infection:** Ingestion of infective eggs.
  - **Laboratory Diagnosis:** Detection of characteristic eggs in stool samples. 
  - **Treatment:** Albendazole, mebendazole, or ivermectin.
  - **Prevention and Control:** Improved sanitation, clean water, proper hygiene, and mass deworming programs. 
  """)

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#Hymenolepsis nana-------------------------------------------------------------------------------------------------------------------------------------------------------------------
  st.header("Hymenolepis nana", anchor="hymenolepis-nana")
  st.header("2.Hymenolepis nana")
  st.subheader("Geographical Distribution")
  st.write("""
  Patients infected with Hymenolepis nana have been found in South America, Southern Europe, Russia, and India. In Thailand, the infection is commonly found in children living in crowded conditions, such as in orphanages and child care institutions.
  """)
  st.subheader("Morphology")
  st.image("https://www.cdc.gov/dpdx/hymenolepiasis/images/2/H_nana_egg_wtmt2.jpg?_=01693")
  st.image("https://ars.els-cdn.com/content/image/1-s2.0-S1198743X15001846-gr1.jpg")
  st.write("""
  - **Egg:** is roughly spherical or ovoid, 30–40 μm in size.
  - **Adult Worm:** H. nana is the smallest intestinal cestode that infects man. It is 5–45 mm in length and less than 1 mm thick. The scolex has 4 suckers and a retractile rostellum with a single row of hooklets.
  """)
  st.subheader("Life cycle")
  st.markdown("""
  The life cycle of the parasite begins when eggs are passed in the stool of an infected individual. These eggs are immediately infective and can survive in the external environment for up to 10 days. Humans become infected primarily through the accidental ingestion of these eggs, often via contaminated food, water, or hands. Once ingested, the eggs hatch in the small intestine, releasing oncospheres. These oncospheres penetrate the intestinal villi and develop into cysticercoid larvae within the tissue.
  As the larvae mature, they return to the intestinal lumen, where they evaginate their scoleces and attach themselves to the intestinal mucosa, particularly in the ileum, a section of the small intestine. There, they develop into adult worms. The adult worms release eggs through gravid proglottids, which are either excreted in the stool or may cause autoinfection within the same host. In cases of autoinfection, the eggs hatch internally, allowing the parasite to complete its life cycle without leaving the host’s body. This mechanism enables the infection to persist for extended periods, sometimes lasting for years.
  """)
  st.image("https://www.cdc.gov/dpdx/hymenolepiasis/modules/H_nana_LifeCycle.gif?_=01675")
  st.markdown("""
  - **Mode of Transmission:** Ingestion of infective eggs from contaminated food or water.
  - **Route of Infection:** Fecal-oral route.
  - **Laboratory Diagnosis:** Detection of eggs in stool samples; ELISA for antigen detection. 
  - **Treatment:** Praziquantel, albendazole, or mebendazole
  - **Prevention and Control:** Improved sanitation, rodent control, and personal hygiene.
  """)

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#diminuta-----------------------------------------------------------------------------------------------------------------------------------------------------------
  st.header("3.Hymenolepis diminuta")
  st.subheader("Geographical Distribution")
  st.write("""
  Hymenolepis diminuta, while less frequent, has been reported from various areas of the world.
  """)
  st.subheader("Morphology")
  st.image("http://www.medical-labs.net/wp-content/uploads/2014/06/H.diminuta-OVA.jpg")
  st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/7/7e/Hymenolepis_diminuta_scolex.jpg/250px-Hymenolepis_diminuta_scolex.jpg")
  st.write("""
  - **Egg:** The egg of Hymenolepis diminuta is generally described as round or slightly oval, measuring approximately 70-86 µm by 60-80 µm
  - **Adult worm:** Adult Hymenolepis diminuta reach 20 to 60 cm, and up to 90 cm
  """)
  st.subheader("Life cycle")
  st.markdown("""
  The life cycle of this parasite begins when eggs are passed in the feces of infected rodents or humans. Gravid proglottids, segments of the adult worm containing eggs, disintegrate within the host’s intestine, releasing eggs that are subsequently excreted in the stool. These eggs are then ingested by arthropod intermediate hosts, such as beetles or fleas, with Tribolium species (flour beetles) being particularly common carriers. Once inside the arthropod, the eggs hatch, and the resulting oncospheres penetrate the intestinal wall, developing into cysticercoid larvae within the insect’s body.
  Humans or rodents may become infected by accidentally ingesting these infected arthropods, often through contaminated food—such as improperly stored cereals—or by exposure in infested environments. When ingested, the cysticercoid larvae are released in the stomach, where they evert their scolex (head) and attach to the intestinal wall using suckers. The worms mature into adults within approximately 20 days. Adult worms, which can grow up to 30 centimeters in length, begin producing eggs that are once again released through the feces, thereby continuing the cycle.
  """)
  st.image("https://www.cdc.gov/dpdx/hymenolepiasis/modules/H_diminuta_LifeCycle.gif?_=01676")
  st.markdown("""
  - **Mode of Transmission:** Ingestion of infected intermediate hosts.
  - **Route of Infection:** Ingestion of infected arthropods.
  - **Laboratory Diagnosis:** Detection of eggs in stool samples.
  - **Treatment:** Praziquantel or niclosamide
  - **Prevention and Control:** Rodent control and proper food handling.
  """)

#--------------------------------------------------------------------------------------------------------------------------------------------------

#taenia-----------------------------------------------------------------------------------------------------------------------------------------------
  st.header("4.Taenia spp.")
  st.subheader("Geographical Distribution")
  st.write("""
  Taenia saginata and T. solium are worldwide in distribution. Taenia solium is more prevalent in poorer communities where humans live in close contact with pigs and eat undercooked pork. Taenia asiatica is limited to Asia and is seen mostly in the Republic of Korea, China, Taiwan, Indonesia, and Thailand.
  """)
  st.subheader("Morphology")
  st.image("https://www.cdc.gov/dpdx/taeniasis/images/1/Taenia_egg.jpg?_=96649")
  st.image("https://cdn.britannica.com/38/130238-004-5120BFB7/Taenia-solium-Scolex-hooks-scolex-wall-tapeworm.jpg")
  st.write("""
  - **Egg:** The eggs measure 30-35 micrometers in diameter and are radially-striated
  - **Adult worm:** Tapeworms (Taenia spp.) are flat, ribbon-like parasites that are white or pale yellow in color. They can grow up to 2–3 meters in length or even longer. The worm consists of a head (scolex) equipped with attachment organs such as suckers and hooks. The body (strobila) is made up of numerous segments called proglottids, each containing both male and female reproductive organs.
  """)
  st.subheader("Life cycle")
  st.markdown("""
  The life cycle of Taenia tapeworms begins when eggs or gravid proglottids are passed in human feces, which can remain viable in the environment for days to months. These are ingested by cattle or pigs through contaminated vegetation—T. saginata infects cattle, while T. solium and T. asiatica infect pigs. Inside the animal, the eggs hatch into oncospheres that migrate to muscle tissue and develop into cysticerci, which can survive for years.Humans become infected by consuming raw or undercooked infected meat. The cysticerci then mature into adult tapeworms in the intestine, a process that takes about two months. The adult worms attach to the small intestine and begin producing proglottids, which are released in feces, each containing tens of thousands of eggs, thus continuing the cycle..
  """)
  st.image("https://www.cdc.gov/dpdx/taeniasis/modules/Taenia_LifeCycle.gif?_=96656")
  st.markdown("""
  - **Mode of Transmission:** Ingestion of undercooked or raw infected meat.
  - **Route of Infection:** Ingestion of cysticerci in meat.
  - **Laboratory Diagnosis:** Detection of eggs or proglottids in stool samples; identification of scolex morphology.
  - **Treatment:** Praziquantel or niclosamide. 
  - **Prevention and Control:** Proper cooking of meat, sanitation, and meat inspection.
  """)

#---------------------------------------------------------------------------------------------------------------------------------------------

#Trichuris trichiura---------------------------------------------------------------------------------------------------------------------------------------------------------
  st.header("5.Trichuris trichiura")
  st.subheader("Geographical Distribution")
  st.write("""
  The third most common round worm of humans. Worldwide, with infections more frequent in areas with tropical weather and poor sanitation practices, and among children. It is estimated that 800 million people are infected worldwide. Trichuriasis occurs in the southern United States.
  """)
  st.subheader("Morphology")
  st.image("https://www.cdc.gov/dpdx/trichuriasis/images/1/Trichuris_trichiura_egg1.jpg?_=01207")
  st.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRj5aWNZL9cDRKC9A839jdlKS6TyAL3930-XQ&s")
  st.write("""
  - **Egg:** Trichuris trichiura eggs are characterized by their barrel or lemon shape, thick shells, and distinctive polar plugs at each end.
  - **Adult worm:** Adult Trichuris trichiura (whipworm) worms are characterized by their whip-like shape, with a long, thin anterior end and a thicker, posterior end
  """)
  st.subheader("Life cycle")
  st.markdown("""
  The life cycle begins when unembryonated eggs are passed in human stool, having been released by adult female worms in the large intestine. In the soil, these eggs develop through several stages and become infective within 15 to 30 days. Humans acquire the infection by ingesting embryonated eggs through contaminated hands, food, or soil.
  Once ingested, the eggs hatch in the small intestine, and the larvae migrate to the large intestine. There, they mature into adult worms, which attach to the mucosal lining of the cecum and colon. About 60 to 70 days after infection, females begin producing 3,000 to 20,000 eggs daily. Adult worms can survive in the host for up to one year.
  """)
  st.image("https://www.cdc.gov/dpdx/trichuriasis/modules/Trichuris_LifeCycle.gif?_=01176")
  st.markdown("""
  - **Mode of Transmission:** Fecal-oral route via ingestion of embryonated eggs from contaminated food or water.
  - **Route of Infection:** Ingestion of infective eggs.
  - **Laboratory Diagnosis:** Detection of characteristic lemon-shaped eggs in stool samples.
  - **Treatment:** Mebendazole, albendazole, or ivermectin.
  - **Prevention and Control:** Improved sanitation, clean water, proper hygiene, and mass deworming programs.
  """)
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#Hookworm---------------------------------------------------------------------------------------------------------------------------------------------------------------------
  st.header("6.Hookworm")
  st.subheader("Geographical Distribution")
  st.write("""
  Hookworm species have a worldwide distribution, mostly in areas with moist, warm climates where larvae can survive in the environment. Both Necator americanus and Ancylostoma duodenale are found in Africa, Asia, Australia and the Americas. Only N. americanus is found in south India and predominates in the Americas, while only A. duodenale is found in the Middle East, North Africa, and northern India.
  """)
  st.subheader("Morphology")
  st.image("https://www.cdc.gov/dpdx/hookworm/images/1/Hookworm_2x2_B.jpg?_=02464")
  st.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQxo5l3jcScXjPG45DCNd9CB-YUUi803OVKpw&s")
  st.write("""
  - **Egg:** Hookworm eggs are typically oval-shaped and have a thin, smooth shell
  - **Adult worm:** Adult hookworms are small, cylindrical, and slightly curved parasites with a hook-like appearance at the front end. They are whitish or grayish in color. Males measure about 8–11 mm long, while females are slightly longer, about 10–13 mm. 
  """)
  st.subheader("Life cycle")
  st.markdown("""
  The life cycle begins when eggs are passed in the stool and hatch into rhabditiform larvae within 1–2 days under warm, moist, and shaded conditions. These larvae develop into infective filariform (L3) larvae within 5–10 days and can survive in soil for several weeks.
  Infective larvae penetrate human skin—often through bare feet—enter the bloodstream, and travel to the lungs. From there, they migrate up the bronchial tree, are swallowed, and reach the small intestine.
  In the jejunum, the larvae mature into adult worms that attach to the intestinal wall and feed on the host's blood. Some larvae may enter a dormant state (hypobiosis). Notably, Ancylostoma duodenale can also infect via oral or transmammary routes, unlike Necator americanus.
  """)
  st.image("https://www.cdc.gov/dpdx/hookworm/modules/Hookworm_LifeCycle_19.jpg?_=81699")
  st.markdown("""
  - **Mode of Transmission:** Transmitted via skin penetration by infective larvae from contaminated soil.
  - **Route of Infection:** Larvae enter through skin → lungs → swallowed → mature in small intestine.
  - **Laboratory Diagnosis:** Stool exam reveals characteristic hookworm eggs under microscope.
  - **Treatment:** Albendazole or mebendazole is effective; iron supplements for anemia.
  - **Prevention and Control:** Wear shoes, improve sanitation, and implement deworming programs.
  """)
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------

#Opisthorchis viverrini ------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  st.header("7.Opisthorchis viverrini")
  st.subheader("Geographical Distribution")
  st.write("""
  Opisthorchis viverrini is found mainly in northeast Thailand, Laos, Cambodia, and central and southern Vietnam. Opisthorchis felineus is found mainly in Italy, Germany, Belarus, Russia, Kazakhstan, and Ukraine.
  """)
  st.subheader("Morphology")
  st.image("https://www.cdc.gov/dpdx/opisthorchiasis/images/1/O_viverrini_egg_wtmt_BAM3.jpg?_=00772")
  st.image("https://www.cdc.gov/dpdx/opisthorchiasis/images/2/O_viverrini_Korea.jpg?_=00767")
  st.write("""
  - **Egg:** The eggs of Opisthorchis viverrini, the Southeast Asian liver fluke, are characterized by their light yellowish-brown, ovoid shape, and a distinct operculum (a lid) at one end
  - **Adult worm:** The adult Opisthorchis viverrini (liver fluke) is a small, flat, and transparent worm, typically measuring about 7 mm in length and 1.5 mm in width. It has a lancet-like shape and is dorso-ventrally flattened
  """)
  st.subheader("Life cycle")
  st.markdown("""
  The life cycle begins when fully developed eggs are passed in the feces of humans or animals. These eggs are ingested by freshwater snails, the first intermediate host, where they hatch into miracidia and develop through several stages—sporocysts, rediae, and then cercariae.
  The cercariae are released from the snails and penetrate freshwater fish, the second intermediate host. Inside the fish, they encyst as metacercariae within the muscles or beneath the scales.
  Humans or other mammals become infected by consuming undercooked or raw fish containing metacercariae. Once ingested, the parasites excyst in the duodenum, migrate to the bile ducts, and mature into adult flukes, beginning egg production within 3 to 4 weeks.
  """)
  st.image("https://www.cdc.gov/dpdx/opisthorchiasis/modules/Opisthorchis_LifeCycle.gif?_=00785")
  st.markdown("""
  - **Mode of Transmission:** Ingestion of undercooked or raw infected fish
  - **Route of Infection:** Ingestion of metacercariae in fish.
  - **Laboratory Diagnosis:** Detection of eggs in stool samples; serological tests for antibodies.
  - **Treatment:**  Praziquantel , albendazole.
  - **Prevention and Control:** Avoid eating raw or undercooked freshwater fish ,Cook fish thoroughly to kill metacercariae ,Mass deworming programs – in high-risk communities to reduce transmission.
  """)
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------

#Minute------------------------------------------------------------------------------------------------------------------------------------------------------------
  st.header("8.Minute intestinal fluke ")
  st.subheader("Geographical Distribution")
  st.write("""
  Egypt, the Middle East, and Far East.
  """)
  st.subheader("Morphology")
  st.image("https://www.troccap.com/wp-content/uploads/2018/09/intestinal-fluke-fig1.jpg")
  st.image("https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEhOMVBlfY8v04JFJRG-t1NvoCfgGAdC9nQuc9FNnfU8mNmb_cj44XyvuIZfOtty0dSZM3lUwTAUJdTbMA9J4ZsHsA3Uz2SEYfLJ65xrCMuqeCgrvnXCu_7EF-05VYvEvtk0qgFCprQBcks/s1600/FASCIOLOPSIS+BUSKI+ADULT+2.jpg")
  st.write("""
  - **Egg:** Minute intestinal fluke (MIF) eggs are small, oval-shaped eggs produced by several species of flukes that infect the small intestine. 
  - **Adult worm:** Minute intestinal flukes are small parasitic flatworms, also known as trematodes, that inhabit the small intestine of humans and other mammals.
  """)
  st.subheader("Life cycle")
  st.markdown("""
  The life cycle begins when embryonated eggs, containing fully developed miracidia, are passed in the feces of infected hosts. These eggs are ingested by specific snails such as Cerithidia or Pironella, where the miracidia hatch and develop through sporocyst, redia, and cercaria stages.
  The cercariae are then released from the snails and encyst as metacercariae in freshwater or brackish fish. Humans and other animals become infected by consuming undercooked or salted fish that contain these metacercariae.
  After ingestion, the metacercariae excyst in the small intestine, attach to the intestinal mucosa, and develop into adult flukes. Definitive hosts include humans, cats, dogs, and fish-eating birds.
  """)
  st.image("https://www.cdc.gov/dpdx/heterophyiasis/modules/Heterophyes_LifeCycle.gif?_=01559")
  st.markdown("""
  - **Mode of Transmission:** Transmitted by ingestion of metacercariae in raw or undercooked freshwater fish.
  - **Route of Infection:** After ingestion, metacercariae excyst in the small intestine and attach to the mucosa.
  - **Laboratory Diagnosis:** Detection of small, operculated eggs in stool via microscopy.
  - **Treatment:** Praziquantel 75 mg/kg/day in 3 divided doses for 1 day.
  - **Prevention and Control:** Avoid eating raw fish, ensure proper cooking, and improve sanitation.
  """)
#-----------------------------------------------------------------------------------------------------------------------
#reference_page----------------------------------------------------------------------------------------------------------------------- 
  st.header("9.Reference ")
  st.markdown("""
  1. Centers for Disease Control and Prevention. (2019, July 19). DPDx - Ascariasis.https://www.cdc.gov/dpdx/ascariasis/index.html
  2. Centers for Disease Control and Prevention. (n.d.). DPDx - Hymenolepiasis.https://www.cdc.gov/dpdx/hymenolepiasis/index.html
  3. Centers for Disease Control and Prevention. (n.d.). DPDx - Hymenolepiasis.https://www.cdc.gov/dpdx/hymenolepiasis/index.html
  4. Centers for Disease Control and Prevention. (n.d.). DPDx - Taeniasis.https://www.cdc.gov/dpdx/taeniasis/index.html
  5. Centers for Disease Control and Prevention. (n.d.). DPDx - Trichuriasis.https://www.cdc.gov/dpdx/trichuriasis/index.html
  6. Centers for Disease Control and Prevention. (n.d.). DPDx - Hookworm.https://www.cdc.gov/dpdx/hookworm/index.html
  7. Centers for Disease Control and Prevention. (n.d.). DPDx - Opisthorchiasis.https://www.cdc.gov/dpdx/opisthorchiasis/index.html
  8. Centers for Disease Control and Prevention. (n.d.). DPDx - Heterophyiasis.https://www.cdc.gov/dpdx/heterophyiasis/index.html
  """)

#--------------------------------------------------------------------------------------------------------



def ai_detector_page():
    st.title("AI Detector")
    st.subheader("Upload & View Image")
    st.write("Upload an image and view it below.")




import streamlit as st
def quiz_page():
    st.write("This is the quiz page")
    
    with st.form(key="quiz_form"):
        q1 = st.radio(
            "Question 1: Which parasite is transmitted by ingestion of undercooked freshwater fish?",
            ["Hymenolepis nana", "Opisthorchis viverrini", "Hookworm", "Minute intestinal flukes"],
        )
        q2 = st.radio(
            "Question 2: Which parasite can auto-infect the host?",
            ["Hymenolepis nana", "Ascaris lumbricoides", "Taenia spp.", "Hookworm"],
        )
        q3 = st.radio(
            "Question 3: Which of the following is the infective stage of Hymenolepis diminuta?",
            ["procercoid", "plerocercoid", "egg", "cysticercoid"],
        )
        q4 = st.radio(
            "Question 4: What helps to differentiate Hymenolepis nana eggs from Hymenolepis diminuta eggs?",
            ["Shell color", "Egg size", "Presence of polar filaments", "Presence of operculum"],
        )
        q5 = st.radio(
            "Question 5: Which parasite egg is the smallest among the flukes listed below?",
            ["Fasciola hepatica", "Opisthorchis viverrini", "Minute intestinal flukes", "Paragonimus westermani"],
        )
        q6 = st.radio(
            "Question 6: What is the infective stage of Ascaris lumbricoides?",
            ["Adult worm", " L2 larva inside the egg", "L1 larva", "Non-embryonated egg"],
        )
        q7 = st.radio(
            "Question 7: Which parasite requires insects as an intermediate host?",
            ["Hymenolepis nana", " Hookworm", " Hymenolepis diminuta", "Trichuris trichiura"],
        )
        q8 = st.radio(
            "Question 8: What distinguishes Taenia solium from T. saginata on microscopic examination of the eggs in stool?",
            ["Shell striations", " Number of uterine branches", "Hooklets on oncosphere"," Cannot be distinguished by egg morphology"],
        )
        q9 = st.radio(
            "Question 9:  In a stool sample, an egg is observed with a thin shell, presence of six hooklets, and no polar filaments. Which parasite is most likely responsible?",
            ["Taenia spp.", " Hymenolepis nana","Trichuris trichiura ","Opisthorchis viverrini"],
        )
        q10 = st.radio(
            "Question 10: How does Hookworm enter the human body?",
            ["Ingested egg ","Inhalation","Skin penetration", "Insect bites"],
        )
        q11 = st.radio(
            "Question 11: What is the infective form of minute intestinal flukes?",
            ["Metacercaria in fish/crustaceans","Egg ingested via water","Cercaria penetrates skin","Miracidium infects snails"],
        )
        q12 = st.radio(
            "Question 12: Which of the following accurately describes the larval migration route of Ascaris lumbricoides in the human host?",
            ["Skin → lymph → intestines","B. Intestine → bloodstream → liver → lungs → pharynx → swallowed ", "Intestine → portal vein → brain ","Lung → bloodstream → skin"],
        )
        q13 = st.radio(
            "Question 13: A stool specimen reveals an egg with a thick, barrel-shaped shell and prominent polar plugs. What is the most likely diagnosis?",
            ["Trichuris trichiura","Ascaris lumbricoides","Hymenolepis nana","Opisthorchis viverrini"],
        )
        q14 = st.radio(
            "Question 14: Which parasite most commonly infects through walking barefoot on soil?",
            ["Hookworm ", "Ascaris ","Trichuris", "Taenia"],
        )
        q15 = st.radio(
            "Question 15: Individuals contract Ascaris lumbricoides via which of the following?",
            ["Inhalation","Insect bite","Ingestion","Inappropriate sexual practices"],
        )
        q16 = st.radio(
            "Question 16:  What is the common name of Taenia solium?",
            ["beef tapeworm", "pork tapeworm", "dwarf tapeworm", "rat tapeworm"],
        )
        q17 = st.radio(
            "Question 17: Which liver fluke species is most commonly found in the northeastern region of Thailand?",
            ["Fasciola hepatica ","Clonorchis sinensis ","Opisthorchis viverrini","Paragonimus westermani"],
        )
        q18 = st.radio(
            "Question 18: Which helminth can penetrate intact skin to infect humans?",
            ["Trichuris trichiura", "Enterobius vermicularis","Necator americanus","Taenia solium"],
        )
        q19 = st.radio(
            "Question 19: Which symptom is least likely in H. diminuta infection?",
            ["Abdominal pain", "Urticaria ","Anemia","Diarrhea"],
        )
        q20 = st.radio(
            "Question 20: What is the usual method of transmission of H. nana in children?",
            ["Inhalation","Waterborne larvae", "Fecal-oral ingestion of eggs ","Transdermal penetration",]
        )
    
        submit_button = st.form_submit_button(label="Submit Answers")

    if submit_button:
        score = 0
        if q1 == "Opisthorchis viverrini":
            score += 1
        if q2 == "Hymenolepis nana":
            score += 1
        if q3.lower() == "cysticercoid":
            score += 1
        if q4 == "Presence of polar filaments":
            score += 1
        if q5.strip() == "Minute intestinal flukes":
            score += 1
        if q6 == "L2 larva inside the egg":
            score += 1
        if q7 == "Hymenolepis diminuta":
            score += 1
        if q8 == "Cannot be distinguished by egg morphology":
            score += 1
        if q9 == " Taenia spp. ":
            score += 1
        if q10 == "Skin penetration":
            score += 1
        if q11 == "Metacercaria in fish/crustaceans":
            score += 1
        if q12 == "Intestine → bloodstream → liver → lungs → pharynx → swallowed":
            score += 1
        if q13 == "Trichuris trichiura":
            score += 1
        if q14 == "Hookworm":
            score += 1
        if q15 == "Ingestion":
            score += 1
        if q16 == "pork tapeworm":
            score += 1
        if q17 == "Opisthorchis viverrini": 
            score += 1
        if q18 == "Necator americanus":
            score += 1
        if q19 == "Anemia": 
            score += 1
        if q20 == "Fecal-oral ingestion of eggs ": 
            score += 1
        
        
            
        st.success(f"Your score is {score} from 16")


        

st.sidebar.markdown("<div class='sidebar-title'>Platform Navigation</div>", unsafe_allow_html=True)
    
pages = {
    "Home": home_page,
    "Parasite Egg Detail": parasite_detail_page,
    "Ai Detector": ai_detector_page,
    "Quiz": quiz_page,
    
    }

selected_page = st.sidebar.selectbox("Select a page", list(pages.keys()))
pages[selected_page]()
