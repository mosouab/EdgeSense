# Script du pitch EdgeSense (3 minutes)

Le texte à dire est le même que dans la vue présentateur (touche `P`).
Entre crochets : ce qu'on fait, pas ce qu'on dit. En **gras** : les mots d'appui,
ceux qu'on peut retrouver même si on perd le fil. `/` = courte pause, `//` = vraie pause.

Conseil du guide « Oser pour innover » : retenir la structure, pas le texte mot à mot.
Trois à cinq répétitions à voix haute suffisent, chrono lancé.

---

## 1 · Titre (0:00 → 0:10)

[Debout, regard sur le jury, une respiration avant de parler.]

> Bonjour, je m'appelle [Prénom]. //
> EdgeSense, c'est un **boîtier** qui fait de la maintenance prédictive **avec les automates qu'une usine a déjà**.

[Clic →]

## 2 · Accroche : Groupe Bel (0:10 → 0:35)

> J'ai fait mon stage chez **Groupe Bel, au Maroc**. / L'usine remontait déjà les données de ses machines et de ses automates vers un serveur Siemens.
>
> Mais ces données servaient à **afficher des courbes**, et c'est tout. //
> [Montrer la carte orange en pointillés.]
> **Aucune alerte** n'en sortait. / Je pense que beaucoup de PME et d'ETI en sont là.

[Clic →]

## 3 · Contexte & Problématique (0:35 → 1:04)

> Les solutions actuelles visent les **grands groupes**. / Augury pose ses propres capteurs et analyse les données chez lui. / Amiral Technologies, soutenue par le CNRS, demande des serveurs et des experts. / Même Tractian, qui vise des usines plus petites, impose ses capteurs et le cloud.
>
> [Montrer le cadre orange en bas à gauche de la carte.]
> Une usine de **taille moyenne** a des techniciens et des automaticiens. / Elle n'a ni équipe data ni ce budget. //
> C'est **pour elle** que je construis EdgeSense.

[Clic →]

## 4 · Concept (1:04 → 1:34)

[Suivre le schéma de la main, de gauche à droite.]

> Concrètement, on **branche le boîtier** sur les automates existants. / Il apprend comment chaque machine se comporte **quand tout va bien**. / Pas besoin d'avoir déjà enregistré des pannes.
>
> Quand une mesure dérive, l'alerte s'affiche sur **l'IHM** que les opérateurs utilisent déjà, avec la cause probable et l'urgence, / et le **ticket** part dans la GMAO. //
> [Montrer la flèche du retour, au-dessus du schéma.]
> Le technicien **confirme ou rejette** l'alerte, et le système en tient compte la fois suivante.

[Clic →]

## 5 · Faisabilité (1:34 → 2:09)

> Le prototype **tourne déjà**. /
> [Montrer les pastilles 1, 2, 3 sur la capture.]
> Ici, elle est en alerte sur la fuite d'air du **5 juin 2020** : la pièce en cause, / l'action à faire, / et le ticket en un clic.
>
> Sur **139 jours** de données réelles du métro de Porto, EdgeSense a signalé **les 4 pannes documentées**, // avec une fausse alerte **tous les dix jours** environ. / Une analyse prend **moins d'une milliseconde** sur un seul cœur de processeur, donc un petit boîtier suffit. / J'estime qu'on est au **TRL 4**.

[Clic →]

## 6 · Impact (2:09 → 2:31)

> Pour l'usine, ça veut dire **moins d'arrêts imprévus**, sans rien installer d'autre que le boîtier. / Le technicien **sait quoi vérifier** avant de se déplacer.
>
> On perd aussi moins de matière et d'énergie, et les données **ne quittent pas l'usine**. //
> Côté modèle économique, j'envisage **un boîtier plus un abonnement** par machine surveillée.

[Clic →]

## 7 · Prochaines étapes & besoins (2:31 → 2:56)

[Ralentir : c'est la demande.]

> La suite, en **trois étapes** : / vingt entretiens avec des responsables maintenance pour valider le besoin, / un pilote sur l'historique d'un vrai site, / puis le branchement en direct sur une ligne.
>
> [Regarder le jury.]
> Pour ça, il me faut **un site pilote**, **un accompagnement** sur le modèle économique et **un premier financement**. //
> EdgeSense, c'est de la maintenance prédictive **pour les usines qui n'ont pas d'équipe data**.

[Clic →]

## 8 · Merci (2:56 → 3:00)

> Merci ! / Je suis à votre disposition pour vos questions.

[Sourire, se taire, attendre la première question.]

---

## La structure à retenir (7 idées)

1. Je suis [Prénom], EdgeSense = maintenance prédictive avec les automates existants.
2. Chez Bel : les données remontaient, aucune alerte n'en sortait.
3. Les offres visent les grands groupes ; les usines moyennes n'ont ni équipe data ni budget.
4. On branche le boîtier, il apprend le normal, alerte sur l'IHM, ticket GMAO, le technicien valide.
5. Ça tourne : 4 pannes sur 4, une fausse alerte tous les dix jours, moins d'une milliseconde, TRL 4.
6. Moins d'arrêts, un technicien qui sait quoi vérifier, un boîtier + abonnement.
7. Entretiens, pilote sur historique, branchement en direct. Il me faut un site, un accompagnement, un financement.

---

## Questions probables du jury

Réponses courtes, puis s'arrêter. Si on ne sait pas : « Je ne l'ai pas encore mesuré, c'est prévu dans le pilote. »

**« Quelle différence avec Tractian ou Augury ? »**
Eux posent leurs propres capteurs et envoient les données dans le cloud. EdgeSense lit les automates déjà en place et tout reste sur site : rien à installer sur les machines, personne à recruter.

**« Vos résultats viennent d'une vraie usine ? »**
Non, pas encore. Ce sont des données réelles et publiques d'un compresseur du métro de Porto, rejouées dans le prototype. C'est pour ça que je parle de TRL 4 et que l'étape suivante est un pilote sur l'historique d'un vrai site.

**« Une fausse alerte tous les dix jours, les techniciens vont l'ignorer ? »**
C'est le risque, d'où le retour du technicien : il rejette l'alerte et le système en tient compte. C'est implémenté dans le prototype ; l'effet reste à mesurer sur de nouvelles données, pendant le pilote.

**« Est-ce que ça marche sur d'autres machines ? »**
Même architecture testée sur un banc hydraulique public : la détection est nette sur certains composants, pas encore sur la vanne, dont le signal est trop rapide pour la fréquence qu'on utilise. Ces résultats sont encore à consolider.

**« Comment ça se connecte aux automates ? »**
Aujourd'hui, le prototype rejoue des données enregistrées. Le connecteur OPC UA / Siemens S7 est la troisième étape du plan.

**« Ça tourne vraiment sur un petit boîtier ? »**
Mesuré sur un seul cœur d'un PC bridé pour imiter un Raspberry Pi, pas encore sur un vrai Pi. Le modèle fait 0,17 Mo et une analyse prend moins d'une milliseconde. Le point à régler est la mémoire du moteur d'exécution actuel (PyTorch) : il faut passer sur un moteur plus léger, type ONNX.

**« Pourquoi ne pas utiliser l'historique des pannes ? »**
Parce que les usines en ont très peu, et rarement bien étiquetées. EdgeSense apprend le fonctionnement normal : il n'a pas besoin d'avoir déjà vu une panne pour en signaler une.

**« Combien ça coûtera ? »**
Je ne fixe pas encore de prix. Le modèle, c'est un boîtier et un abonnement par machine ; les entretiens servent justement à trouver le prix qu'une usine moyenne accepte.

**« Groupe Bel est partenaire ? »**
Non, c'est là que j'ai vu le problème pendant mon stage. [À adapter si vous avez un contact sur place.]

**« Qu'est-ce qui est vraiment innovant ? Une protection PI ? »**
La combinaison : apprentissage sans historique de pannes, un modèle assez léger pour tourner sur site, une alerte qui dit quel capteur regarder et quoi faire, et un système qui apprend des retours du technicien. Pas encore de stratégie PI ; j'aimerais en parler avec la SATT Paris-Saclay.

**« Vous êtes seul sur le projet ? »**
[À compléter : équipe, compétences, qui vous cherchez à recruter.]

---

## Phrase-pitch (60 à 90 s, canevas « Oser pour innover »)

Pour un couloir, un networking ou si on vous demande « en une minute » :

> Saviez-vous que chez Groupe Bel, au Maroc, les machines envoyaient leurs données vers un serveur central sans qu'aucune alerte n'en sorte ? Je développe EdgeSense, un boîtier qui se branche sur les automates existants, pour les équipes de maintenance des usines de taille moyenne. Il apprend le fonctionnement normal de chaque machine, sans historique de pannes, et prévient le technicien avant la panne. Tout tourne sur site. Je cherche une usine partenaire et un accompagnement pour lancer un pilote. Merci.
