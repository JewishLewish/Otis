<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/BlossomDNS/register">
    <img src="https://cdn.discordapp.com/attachments/660227834500874276/1175310288212463706/47._Big_Tree_1.png?ex=656ac400&is=65584f00&hm=0518b63834cd0da8208e79c1b019fd41e170aaa860d4812695fb8a43d43abc55&" alt="Logo" width="200" height="200">
  </a>

  <h3 align="center">Otis Anti-Spam AI</h3>

  <p align="center">
    Go Away Spam!
    <br />
    <a href="https://huggingface.co/Titeiiko/OTIS-Official-Spam-Model"><strong>» » Hugging Face</strong></a>  
    <br />
    <a href="https://github.com/BlossomDNS/register"><strong>» » Github</strong></a>  
    <br />
    <div align="center">
	
![GitHub forks](https://img.shields.io/github/forks/JewishLewish/otis?color=63C9A4&style=for-the-badge)
![GitHub Repo stars](https://img.shields.io/github/stars/JewishLewish/otis?color=63C9A4&style=for-the-badge)
![GitHub](https://img.shields.io/github/license/JewishLewish/otis?color=63C9A4&style=for-the-badge)
![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/JewishLewish/otis?color=63C9A4&style=for-the-badge)

</div>

  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#Quickstart">Quickstart</a>
    </li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>

<!-- Quickstar -->
## Quickstart
```py
# pip install transformers
from transformers import pipeline


def analyze_output(input: str):
    pipe = pipeline("text-classification", model="Titeiiko/OTIS-Official-Spam-Model")
    x = pipe(input)[0]
    if x["label"] == "LABEL_0":
        return {"type":"Not Spam", "probability":x["score"]}
    else:
        return {"type":"Spam", "probability":x["score"]}
    

print(analyze_output("Cһeck out our amazinɡ bооѕting serviсe ѡhere you can get to Leveӏ 3 for 3 montһs for just 20 USD."))

#Output: {'type': 'Spam', 'probability': 0.9996588230133057}
```


<!-- ABOUT THE PROJECT -->
## About The Project


Introducing Otis: Otis is an advanced anti-spam artificial intelligence model designed to mitigate and combat the proliferation of unwanted and malicious content within digital communication channels.


<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b JewishLewish/Otis`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeatures'`)
4. Push to the Branch (`git push origin JewishLewish/Otis`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- LICENSE -->
## License

Distributed under the BSD-3 License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

My Email: lenny@lunes.host

<p align="right">(<a href="#readme-top">back to top</a>)</p>


# OtisV1

```
{'loss': 0.2879, 'learning_rate': 4.75e-05, 'epoch': 0.5}
{'loss': 0.1868, 'learning_rate': 4.5e-05, 'epoch': 1.0}                                                                                                                                                                                                                                                                     
{'eval_loss': 0.23244266211986542, 'eval_runtime': 4.2923, 'eval_samples_per_second': 465.951, 'eval_steps_per_second': 58.244, 'epoch': 1.0}                                                                                                                                                                                
{'loss': 0.1462, 'learning_rate': 4.25e-05, 'epoch': 1.5}                                                                                                                                                                                                                                                                    
{'loss': 0.1244, 'learning_rate': 4e-05, 'epoch': 2.0}
{'eval_loss': 0.19869782030582428, 'eval_runtime': 4.5759, 'eval_samples_per_second': 437.075, 'eval_steps_per_second': 54.634, 'epoch': 2.0}                                                                                                                                                                                
{'loss': 0.0962, 'learning_rate': 3.7500000000000003e-05, 'epoch': 2.5}                                                                                                                                                                                                                                                      
{'loss': 0.07, 'learning_rate': 3.5e-05, 'epoch': 3.0}
{'eval_loss': 0.18761929869651794, 'eval_runtime': 4.1205, 'eval_samples_per_second': 485.372, 'eval_steps_per_second': 60.672, 'epoch': 3.0}                                                                                                                                                                                
{'loss': 0.0553, 'learning_rate': 3.2500000000000004e-05, 'epoch': 3.5}                                                                                                                                                                                                                                                      
{'loss': 0.0721, 'learning_rate': 3e-05, 'epoch': 4.0}
{'eval_loss': 0.19852963089942932, 'eval_runtime': 3.992, 'eval_samples_per_second': 501.004, 'eval_steps_per_second': 62.625, 'epoch': 4.0}                                                                                                                                                                                 
{'loss': 0.0447, 'learning_rate': 2.7500000000000004e-05, 'epoch': 4.5}                                                                                                                                                                                                                                                      
{'loss': 0.0461, 'learning_rate': 2.5e-05, 'epoch': 5.0}
{'eval_loss': 0.20028768479824066, 'eval_runtime': 3.8479, 'eval_samples_per_second': 519.766, 'eval_steps_per_second': 64.971, 'epoch': 5.0}                                                                                                                                                                                
{'loss': 0.0432, 'learning_rate': 2.25e-05, 'epoch': 5.5}                                                                                                                                                                                                                                                                    
{'loss': 0.033, 'learning_rate': 2e-05, 'epoch': 6.0}
{'eval_loss': 0.20464178919792175, 'eval_runtime': 3.9167, 'eval_samples_per_second': 510.638, 'eval_steps_per_second': 63.83, 'epoch': 6.0}                                                                                                                                                                                 
{'loss': 0.0356, 'learning_rate': 1.75e-05, 'epoch': 6.5}                                                                                                                                                                                                                                                                    
{'loss': 0.027, 'learning_rate': 1.5e-05, 'epoch': 7.0}
{'eval_loss': 0.20742492377758026, 'eval_runtime': 3.9716, 'eval_samples_per_second': 503.578, 'eval_steps_per_second': 62.947, 'epoch': 7.0}                                                                                                                                                                                
{'loss': 0.0225, 'learning_rate': 1.25e-05, 'epoch': 7.5}                                                                                                                                                                                                                                                                    
{'loss': 0.0329, 'learning_rate': 1e-05, 'epoch': 8.0}
{'eval_loss': 0.20604351162910461, 'eval_runtime': 4.0244, 'eval_samples_per_second': 496.964, 'eval_steps_per_second': 62.12, 'epoch': 8.0}                                                                                                                                                                                 
{'loss': 0.0221, 'learning_rate': 7.5e-06, 'epoch': 8.5}                                                                                                                                                                                                                                                                     
{'loss': 0.0127, 'learning_rate': 5e-06, 'epoch': 9.0}
{'eval_loss': 0.21241146326065063, 'eval_runtime': 3.9242, 'eval_samples_per_second': 509.659, 'eval_steps_per_second': 63.707, 'epoch': 9.0}                                                                                                                                                                                
{'loss': 0.0202, 'learning_rate': 2.5e-06, 'epoch': 9.5}                                                                                                                                                                                                                                                                     
{'loss': 0.0229, 'learning_rate': 0.0, 'epoch': 10.0}
{'eval_loss': 0.2140526920557022, 'eval_runtime': 3.9546, 'eval_samples_per_second': 505.743, 'eval_steps_per_second': 63.218, 'epoch': 10.0}                                                                                                                                                                                
{'train_runtime': 667.0781, 'train_samples_per_second': 119.926, 'train_steps_per_second': 14.991, 'train_loss': 0.07010261821746826, 'epoch': 10.0} 
```