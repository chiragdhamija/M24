# Installing the dependencies
For a clean installation, initialize a virtual environment
```bash
python -m venv venv
source venv/bin/activate
```
Run the following command to install all the dependencies
```bash
pip install -r requirements.txt
```
To install the node dependencies, navigate to the `cyclone` directory and run `npm install`.
```bash
cd cyclone
npm install
```

# Initializing the application
In the home directory, run `server.py`, and in `cyclone` directory, run `npm start`
```bash
python3 server.py
cd cyclone
npm start
```
