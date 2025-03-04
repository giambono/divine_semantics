import xml.etree.ElementTree as ET

def print_structure(element, indent=0):
    print('  ' * indent + f"{element.tag} {element.attrib}")
    for child in element:
        print_structure(child, indent+1)

# Parse the XML file
tree = ET.parse('commedia-amp.xml')
root = tree.getroot()

# Extract all values of the 'who' attribute
who_values = [elem.attrib['who'] for elem in root.iter() if 'who' in elem.attrib]

whos = list(set(who_values))

print(whos)

# Find all elements with a 'who' attribute (e.g., <q>)
elements_with_who = root.findall(".//*[@who]")

# For each such element, get its descendant <cl> elements
cl_from_who_elements = []
for elem in elements_with_who:
    cl_from_who_elements.extend(elem.findall(".//cl"))

# Print out each matching <cl> element
for cl in cl_from_who_elements:
    ET.dump(cl)