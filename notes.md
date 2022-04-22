use flask to recevie files.
run server cron job to automatically merge the weights, based on timestamp
set up databsae for storing files? --- i'll use a temporary folder for persistience based on day/timestamp, lol

https://leifengblog.net/blog/deploy-flask-applications-on-azure-vps/ -- used this for deploying flask app

- insert auth step?
- client app requests dataset from server 
- device info queried for compute ability
- given dataset, model initiates training loop
    - if interrupted, save current checkpoint (is this even doable?)
- after training loop, upload weights to server (insert auth step 2?) 
  - check timestamp if it was too late, or too early?


serverside:
- serve datasets, 
- midnight cron job to merge the weights by averaging them
- track all clients, reject any stragglers? (or is this the proximal term that people are talking abotu?)


[NUMPY CANNOT BE USED FROM A SUB-INTERPRETER](https://code.google.com/archive/p/modwsgi/wikis/ApplicationIssues.wiki#Python_Simplified_GIL_State_API)
- use [this](https://modwsgi.readthedocs.io/en/master/configuration-directives/WSGIApplicationGroup.html) to resolve --> set to %{GLOBAL}
- go to /etc/apache2/sites-available/{app-name}.conf, and then do the needful

currently using root user to install python packages --- make sure you figure out how to change the mod_wsgi interpreter in config so that you no longer need to do this (i.e. you can use virtualenvs)