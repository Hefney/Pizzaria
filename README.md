<a name="readme-top"></a>

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]

[contributors-shield]: https://img.shields.io/github/contributors/AbdoWise-z/Discordo?style=for-the-badge
[contributors-url]: https://github.com/Hefney/Pizzaria/graphs/contributors

[forks-shield]: https://img.shields.io/github/forks/AbdoWise-z/Discordo?style=for-the-badge
[forks-url]: https://github.com/Hefney/Pizzaria/network/members

[stars-shield]: https://img.shields.io/github/stars/AbdoWise-z/Discordo?style=for-the-badge
[stars-url]: https://github.com/Hefney/Pizzaria/stargazers

[issues-shield]: https://img.shields.io/github/issues/AbdoWise-z/Discordo?style=for-the-badge
[issues-url]: https://github.com/Hefney/Pizzaria/issues

[license-shield]: https://img.shields.io/github/license/AbdoWise-z/Discordo?style=for-the-badge
[license-url]: https://github.com/Hefney/Pizzaria/blob/master/LICENSE.txt

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/Hefney/Pizzaria">
    <img src="images/logo.png" alt="Logo" height="180" style="border-radius: 2%;rotate: 180deg;">
  </a>

<h3 align="center">Pizzaria</h3>
  <p align="center">
    A NLP pizza order semantic parser processor.
    <br />
    <a href="https://github.com/Hefney/Pizzaria/">Explore</a>
    ·
    <a href="https://github.com/Hefney/Pizzaria/tree/main/images">Images</a>
    ·
    <a href="https://github.com/Hefney/Pizzaria/issues">Request Feature</a>
  </p>
</div>

<!-- TABLE OF CONTENTS -->
## Table of Contents
<ol>
  <li>
    <a href="#about-the-project">About The Project</a>
    <ul>
      <li><a href="#built-with">Built With</a></li>
    </ul>
  </li>
  <li>
    <a href="#structure">Structure</a>
  </li>
  <li>
    <a href="#getting-started">Getting Started</a>
  </li>
  <li><a href="#gallery">Gallery</a></li>
  <li><a href="#contributing">Contributing</a></li>
  <li><a href="#license">License</a></li>
  <li><a href="#contact">Contact</a></li>
</ol>

<!-- ABOUT THE PROJECT -->
## About The Project
This is our 4th year (1st term) NLP project. Winning the first place in the competition held in our
department achieving ~80% accuracy in the university final test set. We Feature a semantic pizza order
parser, that given a pizza order input written in natural text it outputs a structured json that 
describes the order. See Example:
```
"Tuan thin crust"
╭──────────────────── JSON ────────────────────╮
│ {                                            │
│     "ORDER": {                               │
│         "PIZZAORDER": [                      │
│             {                                │
│                 "NUMBER": null,              │
│                 "SIZE": null,                │
│                 "STYLE": [                   │
│                     {                        │
│                         "NOT": false,        │
│                         "TYPE": "thin crust" │
│                     }                        │
│                 ],                           │
│                 "AllTopping": []             │
│             }                                │
│         ],                                   │
│         "DRINKORDER": []                     │
│     }                                        │
│ }                                            │
╰──────────────────────────────────────────────╯
```
Other examples can be found at the end of this readme file.

### Built With

The Project was mainly built using python and Jupyter Notebook. We also used C++ to do some data
pre-processing simply because it would be faster that way.

We use libraries such as:
- Cuda
- torch
- Numpy
- nltk
- pickle

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Structure
The project is divided into different layers as follows:
</br>


#### Input Pre-processing
When a query (user order) is received, the input text is corrected of any spelling mistakes that we are 
able to correct, this is done following the 3 strategies in order:
1. Look for **one** missing letter in each word
2. Try to find the nearest word using "language-tool-python"
3. Try to find the neatest word using "pyspellchecker"
4. return a <unk>
If a word is not inside our training vocab we assume that it's spelling is in-correct.

#### Model inference
We follow a layered structure as follows:

##### Layer 1
we have one model that tries the predict where a pizza order is and where a drink order is.

##### Layer 2
We have two models. the first one tries to parse a pizza order given the output of the model in layer 1, 
the second does the same but for a drink order. 

##### Layer 3
If a pizza model in layer 2, outputs a region as a "Complex Topping", a model in layer 3 is responsible
for parsing that topping.

in summary, we have the following structure for the models:
```
Layer 1
 ├── Model 1: Predicts Pizza Order Region
 └── Model 2: Predicts Drink Order Region
    ↓
Layer 2
 ├── Model 1: Parses Pizza Order (using Layer 1's output)
 └── Model 2: Parses Drink Order (using Layer 1's output)
    ↓
Layer 3 (Triggered by complex toppings)
 └── Model 1: Parses Complex Pizza Topping (if pizza model in Layer 2 outputs "Complex Topping")
```

This is achieved by Seq2Seq mapping, using a LSTM for each model.


<!-- GETTING STARTED -->
## Getting Started
Before you begin, ensure you have the following installed:

- [Node.js](https://nodejs.org/) (v14.x or later)
- [npm](https://www.npmjs.com/) (v6.x or later) or [Yarn](https://yarnpkg.com/) (v1.22.x or later)
- A code editor, such as [Visual Studio Code](https://code.visualstudio.com/)

#### Steps
1. Clone the Repository
```bash
git clone https://github.com/Hefney/Pizzaria.git
cd Pizzaria
```
2. Install Dependencies

Install the necessary dependencies using npm:
```bash
pip install language-tool-python pyspellchecker pyenchant
```
you will also need to install CUDA with torch, head to pytorch website for more info.
3. Training

The entire train pipeline can be found in [model_trainer.ipynb](model_trainer.ipynb),
You will need to prepare the input dataset by running the C++ code found in [input_parser](input_parser),
Then you run the train pipeline for each model separately.

4. Running

We already provide the models that we submitting in the compilation (can be found in [live](live)). to run them
just follow the cells found in [query_runner.ipynb](query_runner.ipynb).

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Gallery
```
"two pizzas with pepperoni and extra chesse"
╭───────────────────── JSON ─────────────────────╮
│ {                                              │
│     "ORDER": {                                 │
│         "PIZZAORDER": [                        │
│             {                                  │
│                 "NUMBER": "two",               │
│                 "SIZE": null,                  │
│                 "STYLE": [],                   │
│                 "AllTopping": [                │
│                     {                          │
│                         "NOT": false,          │
│                         "QUANTITY": null,      │
│                         "TOPPING": "pepperoni" │
│                     },                         │
│                     {                          │
│                         "NOT": false,          │
│                         "QUANTITY": "extra",   │
│                         "TOPPING": "chesse"    │
│                     }                          │
│                 ]                              │
│             }                                  │
│         ],                                     │
│         "DRINKORDER": []                       │
│     }                                          │
│ }                                              │
╰────────────────────────────────────────────────╯
```


```
"two large pizzas and lemon juice"
╭─────────────────── JSON ───────────────────╮
│ {                                          │
│     "ORDER": {                             │
│         "PIZZAORDER": [                    │
│             {                              │
│                 "NUMBER": "two",           │
│                 "SIZE": "large",           │
│                 "STYLE": [],               │
│                 "AllTopping": []           │
│             }                              │
│         ],                                 │
│         "DRINKORDER": [                    │
│             {                              │
│                 "DRINKTYPE": "lemon juice" │
│             }                              │
│         ]                                  │
│     }                                      │
│ }                                          │
╰────────────────────────────────────────────╯
```


```
"a cup of water"
╭──────────────── JSON ────────────────╮
│ {                                    │
│     "ORDER": {                       │
│         "PIZZAORDER": [],            │
│         "DRINKORDER": [              │
│             {                        │
│                 "NUMBER": "a",       │
│                 "DRINKTYPE": "water" │
│             }                        │
│         ]                            │
│     }                                │
│ }                                    │
╰──────────────────────────────────────╯
```

```
"two pizzas with peppers and one pizza with pepperoni large two cokes"
╭───────────────────── JSON ─────────────────────╮
│ {                                              │
│     "ORDER": {                                 │
│         "PIZZAORDER": [                        │
│             {                                  │
│                 "NUMBER": "two",               │
│                 "SIZE": null,                  │
│                 "STYLE": [],                   │
│                 "AllTopping": [                │
│                     {                          │
│                         "NOT": false,          │
│                         "QUANTITY": null,      │
│                         "TOPPING": "peppers"   │
│                     }                          │
│                 ]                              │
│             },                                 │
│             {                                  │
│                 "NUMBER": "one",               │
│                 "SIZE": null,                  │
│                 "STYLE": [],                   │
│                 "AllTopping": [                │
│                     {                          │
│                         "NOT": false,          │
│                         "QUANTITY": null,      │
│                         "TOPPING": "pepperoni" │
│                     }                          │
│                 ]                              │
│             }                                  │
│         ],                                     │
│         "DRINKORDER": [                        │
│             {                                  │
│                 "NUMBER": "two",               │
│                 "DRINKTYPE": "cokes"           │
│             }                                  │
│         ]                                      │
│     }                                          │
│ }                                              │
╰────────────────────────────────────────────────╯
```

The model isn't perfect. but its decent :)
<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

This project was developed by
<table>
<tr>

<td align="center">
<a href="https://github.com/AbdoWise-z" target="_black">
<img src="https://avatars.githubusercontent.com/u/82497296?v=4" width="150px;" alt="Avatar" style="border-radius: 50%;"/><br /><sub><b>Abdulrahman Mohammed</b></sub> <br/></a>
</td>

<td align="center">
<a href="https://github.com/Elkapeer" target="_black">
<img src="https://avatars.githubusercontent.com/u/87075455?v=4" width="150px;" alt="Avatar" style="border-radius: 50%;"/><br /><sub><b>Osama Saleh</b></sub><br/></a>
</td>

<td align="center">
<a href="https://github.com/amr-salahuddin" target="_black">
<img src="https://avatars.githubusercontent.com/u/120669828?v=4" width="150px;" alt="Avatar" style="border-radius: 50%;"/><br /><sub><b>Amr Salahuddin</b></sub><br/></a>
</td>

<td align="center">
<a href="https://github.com/Hefney" target="_black">
<img src="https://avatars.githubusercontent.com/u/96011550?v=4" width="150px;" alt="Avatar" style="border-radius: 50%;"/><br /><sub><b>Abdulrahman Hefney</b></sub><br/></a>
</td>

</tr>
 </table>

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTACT -->
## Contact

[Abdulrahman Mohammed Abdulfattah](https://www.linkedin.com/in/abdo-mohamed-5b3506252/) - <moh2002.abdow@gmail.com>

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!--difiniations -->