// Run through create_qupath_project.py. Uses QuPath's public project/server APIs.
import qupath.lib.io.GsonTools
import java.awt.image.BufferedImage
import qupath.lib.images.ImageData
import qupath.lib.images.servers.ImageServerProvider
import qupath.lib.projects.Projects
import qupath.lib.regions.RegionRequest

if (args.length != 2) throw new IllegalArgumentException('Expected roster JSON and destination directory')
def rows = GsonTools.getInstance().fromJson(new File(args[0]).text, List.class)
if (rows.isEmpty() || rows.size() > 100) throw new IllegalArgumentException('Expected 1-100 slides')
def destination = new File(args[1])
if (destination.exists()) throw new IllegalArgumentException('Refusing to overwrite an existing project')
if (!destination.mkdirs()) throw new IOException('Cannot create project directory')
def project = Projects.createProject(destination, BufferedImage.class)
rows.each { row ->
    def server = ImageServerProvider.buildServer(row.absolute_path as String, BufferedImage.class)
    try {
        def entry = project.addImage(server.getBuilder())
        def stain = (row.stain ?: '').toString()
        entry.setImageName("${row.label ?: 'unlabelled'} | ${row.case_submitter_id}" + (stain ? " | ${stain}" : ''))
        row.each { key, value ->
            if (!(key in ['path', 'absolute_path']) && value != null && value.toString())
                entry.putMetadataValue(key as String, value.toString())
        }
        def normalizedStain = stain.toUpperCase(Locale.ROOT).replaceAll('[^A-Z]', '')
        boolean isHE = normalizedStain in ['HE', 'HESTAIN', 'HEMATOXYLINANDEOSIN'] ||
            (!stain && (row.project ?: '').startsWith('TCGA-'))
        def imageData = new ImageData(server, isHE ? ImageData.ImageType.BRIGHTFIELD_H_E : ImageData.ImageType.BRIGHTFIELD_OTHER)
        entry.saveImageData(imageData)
        double downsample = Math.max(1.0, Math.max(server.getWidth(), server.getHeight()) / 512.0)
        def request = RegionRequest.createInstance(server.getPath(), downsample, 0, 0, server.getWidth(), server.getHeight())
        entry.setThumbnail(server.readRegion(request))
        println "Added ${entry.getImageName()}: ${server.getWidth()} x ${server.getHeight()}"
    } finally {
        server.close()
    }
}
project.syncChanges()
if (project.getImageList().size() != rows.size()) throw new IllegalStateException('Project image count mismatch')
println "Saved ${project.getPath()} with ${rows.size()} referenced slides"
