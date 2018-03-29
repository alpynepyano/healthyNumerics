Title: A Python Website & Blog with Pelican and Jupyter
Date: 2017-08-06 21:30
Category: devTec
Tags: Pelican, Jupyter, Python
Slug: python-pelican-website-with-jupyter-part-1
Authors: Peter Schuhmacher
Cover: /p6001/img6001/a1_fractal_001.png
Summary: A short guide how to create a Pelican website for blogs with Jupyter

This post is rather an **overview** than a detailled tutorial. It summarizes the different steps you have to take to run a Pelican website where you use Jupyter to write the blogs.

Ressource #1 is always **http://docs.getpelican.com/en/stable/**

Here some other interesting sources:  http://marpat.github.io/python-anaconda-and-pelican-on-windows.html and http://nafiulis.me/making-a-static-blog-with-pelican.html

### The overall Workflow
This is a general overview over all steps needed until you can run your Pelican website in your local browser. The sections below give some details to the single steps.

0. Install the pelican software following http://docs.getpelican.com/en/stable/
1. `!pelican-quickstart`    (starts a new pelican project)
1. modify **`pelicanconf.py`** to choose a **theme** you like
2. **write an article** with Jupyter (see below)
2. from Jupyter export your article as `myArticleName.md` or `myArticleName.rst`
2. edit `myArticleName.md` and delete the blanc lines at the top of the files
2. deposit your articles in directory `content` as `myArticleName.md` or `myArticleName.rst`
3. Run pelican to transform the content into HTML with`!pelican content -s my_pelicanconf.py -t /projects/your-site/themes/your-theme` ; or simpler: `!pelican content` if your configuration file has the default name `pelicanconf.py`
4. change to the folder where te output is: `!cd ~/projects/yoursite/output`
5. run your local web server `!python -m pelican.server`   ; or alternatively: `!python -m http.server` 
6. start your browser with    `http://localhost:8000/` to see the created website

### Appearence, themes, pelicanconf.py
The `pelicanconf.py` file is the controlling configuration file of your perlican website. There are some dozens of prepared themes you can download. Each theme comes with two directories (`static`, `templates`). In `pelicanconf.py` you have to set the path to your choosen theme. It might be helpful that you have a  `pelicanconf_myTheme.py` for each of our prefered theme.

1. download the pelican themes from http://www.pelicanthemes.com/
2. place your favorite theme in in your pelican directory and set in  `pelicanconf_myTheme.py` the path and the name. e.g.  `THEME = 'themes/myTheme'`

### Write an Article
1. Use Jupyter to write your article
1. Write into the first Jupyter cell of your article the pelican head:

        Title: My everything telling title
        Date: 2018-08-03 10:20
        Modified: 2018-08-15 19:30
        Category: Python
        Tags: pelican, publishing
        Slug: my-universal-post
        Authors: John Devil, Lydia Angel
        Cover: /p001/img001/pic1.png
        Summary: All you want to know about pelican

2. export/transform the Jupyter file to a Markdown file (or to a rST file) with `file -> download as`
3. edit the Markdown file and delete the trailing blank line at the top
4. if not already done: deposit the Markdown file in the `content` directory

### Structure of the Content directory and Categories
##### Variante: `USE_FOLDER_AS_CATEGORY = True`
You can use the content directory as a classifier which assigns the articles to a category. For that in  `pelicanconf.py` you must set `USE_FOLDER_AS_CATEGORY = True`.  That means that *Politics*, *QuantsNumerics* and *Technics* will appear as categories in your pelican website.

    website/
    ├── content
    │   ├── Politics/
    │   │   └── pArticle01.md
    │   │   └── pArticle02.md
    │   ├── QuantsNumerics/
    │   │   └── qArticle01.md
    │   │   └── qArticle02.md    
    │   ├── Technics/
    │   │   └── tArticle01.md
    │   │   └── tArticle02.md
    │   ├── article20.md
    │   ├── article21.md
    │   └── pages
    │       └── AboutUs.md
    │       └── furtherProjects.md
    └── pelicanconf.py


##### Variante: `USE_FOLDER_AS_CATEGORY = False`
In the head section of your article you can give explicitely the article a category with 
        `Category: Python`
You can now structure your content directory as you prefer it, e.g. a directory for each article which is helpful if you have pictures and other static files which belong to the article:

    website/
    ├── content
    │   ├── p001/
    │   │   └── pArticle001.md
    │   │   └── img001
    │   │       ├── pic1.png
    │   │       ├── pic2.png
    │   │       └── movie.mp4
    │   ├── p002/
    │   │   └── pArticle002.md
    │   │   └── img002
    │   │       ├── pic1.png
    │   │       └── pic2.png
    │   └── pages
    │       └── AboutUs.md
    │       └── furtherProjects.md
    └── pelicanconf.py

In  `pelicanconf.py` you must set 

      STATIC_PATHS  = ['p001','p002',]

### How to include Images into the Article

For detaiiled instructions follow [this](http://docs.getpelican.com/en/3.6.3/content.html#attaching-static-files):


In `pelicanconfig.py` set:

    PATH = 'content'
    STATIC_PATHS = ['p001','p002']
    ARTICLE_PATHS = STATIC_PATHS

Open the `md`-file and have a look where the pictures are. You will find something like `output_3_0.png`. Replace that by

    ![pic_A1]({attach}img001/myPic1.png)
    ![pic_A2]({attach}img001/myPic2.png)

and

    ![pic_B1]({attach}img002/myPic1.png)
    ![pic_B2]({attach}img002/myPic2.png)



### Add an Image to the Summary

In the Jupyter head section, `cover` is a picture that is added to the summary in the table of content.

- if you have images in your article, use
        Cover: /posts/img001/pic1.png
- if do not have images in your article, use
        Cover: /p001/img001/pic1.png
        
Note that $posts$ has to correspond to the setting  in _myPelicanConf.py _:

    ARTICLE_URL = 'posts/{slug}.html'
    
and $p001$ corresponds to the structure of your content directory

In order to include the cover, make an extension there where `article.summary` already is used. I use pelican-theme **alchemy**  (that uses bootstrap) and  `article.summary` is in `index.html`. I extended it as follows: 

      <div class="col-sm-8">
        <h4 class="title"><a href="{{ SITEURL }}/{{ article.url }}">{{ article.title }}</a></h4>
        <div class="content">
          {{ article.summary|striptags }}
    	  {% if article.cover %}	  	   
             <div class="container"><div class="col-md-6" style="padding-left: 0px;  padding-right: 0px;">
               <img src="{{ article.cover }}" class="img-fluid">
               </div></div>	  
    	  {% endif %} 
        </div>
      </div>


### Pages

If you create a folder named `pages` inside the content folder, all the files in it will be used to generate static pages, such as _About_ or _Contact_ pages.

You can use the *DISPLAY_PAGES_ON_MENU* setting in `pelicanconfig.py` to control whether all those pages are displayed in the primary navigation menu. (Default is True.)

If you want to exclude any pages from being linked to or listed in the menu then add a _status: hidden_ attribute to its metadata. This is useful for things like making error pages that fit the generated theme of your site.


### Development or production state
During the __development__ we set in _myPelicanConf.py _

    #--- development stage ----------------
    SITEURL = 'http://localhost:8000' 
    LOAD_CONTENT_CACHE = False

Comment out these lines when you want to pblish your site (__production__) and set in _myPelicanConf.py_ instead

    #------ production stage ------------------
    SITEURL = 'https://www.myDomainName.com'
    RELATIVE_URLS = False

### Run your Pelican Website 

#### with command line (CMD window)

1. change to your Pelican project as your current directory 
2. let Pelican transform your content into output with
   2. ```pelican content```, if your setting file is named as `pelicanconf.py`
   2. ```pelican /path/to/your/content/ -s path/to/your/myPelicanConf.py``` in the general case
3. run a webserver with ```python -m http.server```
4. start in your browser http://localhost:8000/ to see the result


#### with Jupyter and  command line (CMD window)

1. change to your Pelican project as your current directory 
2. in Jupyter run
   2. ```!pelican content```, if your setting file is named as `pelicanconf.py`
   2. ```!pelican -s path/to/your/myPelicanConf.py``` in the general case
3. have a CMD window open with running a webserver permanentely with ```python -m http.server```
4. refresh/reload your browser http://localhost:8000/ to see the result


    pelican pyanoB/content -s pyanoB/pelicanconf_alchemy.py
    
    pelican content -s pelicanconf_alchemy.py
    
    cd C:\path\to\myPelicanDirectory\output

    python -m http.server

    http://localhost:8000/


### Deployment
--> see an other post for
- github as publication server
- Discus as an added discussion plattform
- fabric3 for further automatisation
