// Run through create_qupath_project.py. Uses QuPath's public project/server APIs.
import qupath.lib.io.GsonTools
import java.awt.image.BufferedImage
import qupath.lib.images.ImageData
import qupath.lib.images.servers.ImageServerProvider
import qupath.lib.projects.Projects
import qupath.lib.regions.RegionRequest

if (args.length != 2) throw new IllegalArgumentException('Expected roster JSON and destination directory')
def rows = GsonTools.getInstance().fromJson(new File(args[0]).text, List.class)
if (rows.size() != 10) throw new IllegalArgumentException('Expected ten slides')
def destination = new File(args[1])
if (destination.exists()) throw new IllegalArgumentException('Refusing to overwrite an existing project')
if (!destination.mkdirs()) throw new IOException('Cannot create project directory')
def project = Projects.createProject(destination, BufferedImage.class)
rows.each { row ->
    def server = ImageServerProvider.buildServer(row.absolute_path as String, BufferedImage.class)
    try {
        def entry = project.addImage(server.getBuilder())
        entry.setImageName("${row.label} | ${row.case_submitter_id}")
        ['label', 'project', 'case_id', 'case_submitter_id', 'site', 'file_id', 'file_name', 'md5sum'].each { key ->
            entry.putMetadataValue(key, row[key] as String)
        }
        def imageData = new ImageData(server, ImageData.ImageType.BRIGHTFIELD_H_E)
        entry.saveImageData(imageData)
        double downsample = Math.max(1.0, Math.max(server.getWidth(), server.getHeight()) / 512.0)
        def request = RegionRequest.createInstance(server.getPath(), downsample, 0, 0, server.getWidth(), server.getHeight())
        entry.setThumbnail(server.readRegion(request))
        println "Added ${row.label} | ${row.case_submitter_id}: ${server.getWidth()} x ${server.getHeight()}"
    } finally {
        server.close()
    }
}
project.syncChanges()
if (project.getImageList().size() != 10) throw new IllegalStateException('Project image count mismatch')
println "Saved ${project.getPath()} with ten referenced slides"
