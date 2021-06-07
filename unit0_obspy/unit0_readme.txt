roses2021 unit0 readme

This is an overview of Obspy prepared by Sydney Dybing for ROSES2020 

environment:
- Use the default 'roses' environment.

notebooks:
- The "follow along" notebook is completed with the lecture video.
- The "full" notebook is a solution and should execute without exceptions.
- the "lab" notebook contains prompts for a student to complete.

data:
- 4 miniseed files are included.

faq/notes:
- Refer to the official Obspy tutorial for more info: docs.obspy.org/tutorial/
- Consider using Python's f-strings when building a string from variables (like nslc codes).
- Obspy's 'get_stations' and 'attach_response' can be used if 'get_waveforms(...,attach_response=True) does not work as expected.
- Obspy's 'read_inventory' function is used to attach response from a local file.

